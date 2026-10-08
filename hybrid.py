"""
ResNet-style convolution + attention hybrid (in the spirit of BoTNet, Srinivas et al. 2021) whose attention
blocks use STR-1 skeleton-graph sparse attention (geometry.py, docs/STR1.md). Trained from scratch, or from ImageNet ResNet-D weights with --pretrained.

    stem   3x3/2 (32) -> 3x3 (32) -> 3x3 (64) -> maxpool/2                     ResNet-D stem, 56x56
    stage1 bottleneck x L1   width  64 -> 256 ch                            56x56   conv
    stage2 bottleneck x L2   width 128 -> 512 ch, stride 2                  28x28   conv
    stage3 bottleneck x L3   width 256 -> 1024 ch, stride 2                 14x14   conv + ATTENTION
    stage4 bottleneck x L4   width 512 -> 2048 ch, stride 2                  7x7    conv + ATTENTION
    head   global average pool -> Linear -> 1 logit

In an attention stage, the first (strided) block stays a conv bottleneck and every later block replaces its
3x3 conv by multi-head self-attention over the feature-map grid (one head per 64 channels) with a learned 2-D
relative position bias:  1x1 conv -> ATTENTION -> 1x1 conv, residual. With attention=str1 the sparse heads
are masked to the skeleton graph of the input image, computed for the 14x14 and 7x7 grids (same tokens as a
ViT-S/16 at 14x14). `hybrid_attn_stages` picks the stages ("3,4" default, "4" = BoTNet placement, "" = a
plain ResNet baseline); attention=mhsa gives the dense BoTNet-style baseline.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from geometry import GeoAttention, SkeletonGraph

# name -> blocks per stage
HYBRID_VARIANTS = {
    "hyb_r26": (2, 2, 3, 2),     # ResNet-26 layout: 3 attention blocks (2 @ 14x14, 1 @ 7x7)
    "hyb_r50": (3, 4, 6, 3),     # ResNet-50 layout: 7 attention blocks (5 @ 14x14, 2 @ 7x7)
}
WIDTHS = (64, 128, 256, 512)


@dataclass
class HybridConfig:
    variant: str = "hyb_r26"
    arch: str = "hybrid"
    image_size: int = 224
    layers: tuple = (2, 2, 3, 2)
    attn_stages: str = "3,4"         # stages (1-4) with attention blocks; "" / "none" = plain ResNet
    head_dim: int = 64               # channels per attention head
    drop_path: float = 0.0
    attention: str = "str1"          # mhsa | str1
    str_k: int = 8
    str_radius: float = 4.0
    str_alpha: float = 30.0
    str_beta: float = 30.0
    str_rho: float = 1.0
    str_mode: str = "soft"
    str_dense_heads: int = 1
    str_coh_gate: bool = True
    str_prior: str = "ridge"
    str_sigma: float = 2.0
    num_outputs: int = 1

    def __post_init__(self):
        self.layers = tuple(self.layers)
        s = "" if self.attn_stages is None else str(self.attn_stages).replace(" ", "")
        if s.lower() in ("none", "null", "0"):     # PowerShell 5.1 drops "" args, so accept these for "no stages"
            s = ""
        self.attn_stages = ",".join(str(int(t)) for t in s.split(",") if t)

    @property
    def stages_with_attn(self):
        return {int(s) for s in self.attn_stages.split(",") if s}

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_variant(cls, variant="hyb_r26", **overrides):
        if variant not in HYBRID_VARIANTS:
            raise ValueError(f"unknown hybrid variant {variant!r}; choose from {', '.join(HYBRID_VARIANTS)}")
        base = dict(variant=variant, layers=HYBRID_VARIANTS[variant])
        base.update({k: v for k, v in overrides.items() if v is not None and k in cls.__dataclass_fields__})
        return cls(**base)


class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        return x * x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep) / keep


def conv_bn(cin, cout, k=1, stride=1, act=True):
    layers = [nn.Conv2d(cin, cout, k, stride, k // 2, bias=False), nn.BatchNorm2d(cout)]
    return nn.Sequential(*layers, nn.ReLU(inplace=True)) if act else nn.Sequential(*layers)


class Bottleneck(nn.Module):
    """1x1 -> (3x3 conv | self-attention) -> 1x1, residual (ResNet-D shortcut)."""

    def __init__(self, cin, width, stride, cfg, grid=None, drop_path=0.0):
        super().__init__()
        cout = width * 4
        self.reduce = conv_bn(cin, width, 1)
        self.grid = grid                                   # set => attention block on a grid x grid map
        if grid:
            self.attn = GeoAttention(width, width // cfg.head_dim, True, 0.0, 0.0, cfg.attention,
                                     cfg.str_dense_heads, cfg.str_mode == "soft", rel_grid=grid)
            self.post = nn.Sequential(nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        else:
            self.conv = conv_bn(width, width, 3, stride)
        self.expand = conv_bn(width, cout, 1, act=False)
        nn.init.zeros_(self.expand[1].weight)              # zero-init last BN: block starts as identity
        self.short = nn.Identity() if stride == 1 and cin == cout else nn.Sequential(
            nn.AvgPool2d(stride) if stride > 1 else nn.Identity(), conv_bn(cin, cout, 1, act=False))
        self.drop_path = DropPath(drop_path)

    def forward(self, x, geo=None):
        y = self.reduce(x)
        if self.grid:
            B, C, H, W = y.shape
            t = self.attn(y.flatten(2).transpose(1, 2), geo.get(H) if geo else None)
            y = self.post(t.transpose(1, 2).reshape(B, C, H, W))
        else:
            y = self.conv(y)
        return torch.relu(self.short(x) + self.drop_path(self.expand(y)))


class HybridNet(nn.Module):
    def __init__(self, cfg: HybridConfig):
        super().__init__()
        self.cfg = c = cfg
        if c.image_size % 32:
            raise ValueError("image_size must be divisible by 32")
        self.stem = nn.Sequential(conv_bn(3, 32, 3, 2), conv_bn(32, 32, 3), conv_bn(32, 64, 3),
                                  nn.MaxPool2d(3, 2, 1))
        total, n = sum(c.layers), 0
        cin, size, grids = 64, c.image_size // 4, set()
        self.stages = nn.ModuleList()
        for si, (L, w) in enumerate(zip(c.layers, WIDTHS), start=1):
            blocks = []
            for b in range(L):
                stride = 2 if (b == 0 and si > 1) else 1
                if stride == 2:
                    size //= 2
                attn = si in c.stages_with_attn and b > 0
                if attn:
                    grids.add(size)
                blocks.append(Bottleneck(cin, w, stride, c, size if attn else None, c.drop_path * n / max(1, total - 1)))
                cin, n = w * 4, n + 1
            self.stages.append(nn.ModuleList(blocks))
        self.head = nn.Linear(cin, c.num_outputs)
        self.graphs = nn.ModuleDict({str(g): SkeletonGraph(g, c.str_k, c.str_radius, c.str_alpha, c.str_beta, c.str_rho,
                                                           c.str_mode, c.str_coh_gate, c.str_prior, c.str_sigma)
                                     for g in sorted(grids)}
                                    ) if c.attention == "str1" else None
        self.grid = c.image_size // 16                      # finest attention grid (Grad-CAM random baseline)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.modules():                              # re-apply after the generic init
            if isinstance(m, Bottleneck):
                nn.init.zeros_(m.expand[1].weight)

    def geometry(self, x):
        """{grid size: STR-1 graph} for the attention grids (None for MHSA / no attention)."""
        return {int(g): G(x) for g, G in self.graphs.items()} if self.graphs else None

    def features(self, x):
        geo = self.geometry(x)
        x = self.stem(x)
        for stage in self.stages:
            for blk in stage:
                x = blk(x, geo)
        return x

    def forward(self, x):
        out = self.head(self.features(x).mean((2, 3)))
        return out.squeeze(-1) if self.cfg.num_outputs == 1 else out

    # ---------------------------------------------------------------- reporting
    @torch.no_grad()
    def param_report(self):
        count = lambda m: sum(p.numel() for p in m.parameters())
        rows = {"stem": count(self.stem)}
        for i, st in enumerate(self.stages, 1):
            rows[f"stage{i}"] = count(st)
        rows["head"] = count(self.head)
        return dict(rows=rows, total_params=sum(p.numel() for p in self.parameters()), tokens=self.grid ** 2)

    def summary(self):
        c, r = self.cfg, self.param_report()
        line = "-" * 62
        out = [line, f"{c.variant} | ResNet-D bottleneck layout {c.layers} | {c.image_size}px",
               f"attention stages: {c.attn_stages or 'none (plain ResNet)'} | {attention_desc(c)}", line]
        size = c.image_size // 4
        for i, (st, (k, v)) in enumerate(zip(self.stages, list(r["rows"].items())[1:-1]), 1):
            size = size // 2 if i > 1 else size
            n_att = sum(1 for b in st if b.grid)
            label = f"{k} {size}x{size} ({len(st)} blocks, {n_att} attention)"
            out.append(f"  {label:<40}{v:>12,}")
        out.insert(4, f"  {'stem':<40}{r['rows']['stem']:>12,}")
        out += [f"  {f'head ({c.num_outputs} logit' + ('s)' if c.num_outputs > 1 else ')'):<40}{r['rows']['head']:>12,}", line,
                f"  {'TOTAL parameters':<40}{r['total_params']:>12,}  ({r['total_params'] / 1e6:.2f}M)", line]
        return "\n".join(out)


def attention_desc(c):
    if not c.attn_stages:
        return "no attention"
    if c.attention != "str1":
        return "MHSA (dense) + relative position bias"
    return (f"STR-1 skeleton-graph ({c.str_mode}, k={c.str_k}, radius={c.str_radius:g}, alpha={c.str_alpha:g}, "
            f"beta={c.str_beta:g}, {c.str_prior} prior, gate {'on' if c.str_coh_gate else 'off'}, "
            f"{c.str_dense_heads} dense head(s)) + relative position bias")
