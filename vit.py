"""
Plain Vision Transformer (Dosovitskiy et al. 2021 / DeiT recipe) for binary SDNET2018 crack classification,
written to be trained from scratch (ImageNet init only with --pretrained, see pretrained.py).

    image -> stem (16x16 patchify, or a small conv stem) -> [CLS] + learned pos-embed
          -> depth x [ LN -> MHSA -> (LayerScale) -> DropPath  +  LN -> MLP -> (LayerScale) -> DropPath ]
          -> LN -> pool ([CLS] or mean of patch tokens) -> Linear -> 1 logit (P(Cracked) = sigmoid)

Variants (``VARIANTS``): vit_s16 is the main model and matches timm's ``vit_small_patch16_224`` exactly
(dim 384, depth 12, 6 heads, MLP 4x, qkv bias; 21,666,049 params with a 1-logit head, state-dict
keys identical apart from patch_embed.proj -> stem.proj). The others are the
same architecture with a different width / depth / patch size, so results are directly comparable.

Every config field can be overridden from the CLI (``--drop-path 0.2 --stem conv --pool gap`` ...).

``attention="str1"`` swaps MHSA for STR-1 skeleton-graph sparse attention (geometry.py, docs/STR1.md): the
graph is built once per image from the input and shared by all blocks; [CLS] stays densely connected.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from geometry import GeoAttention, SkeletonGraph, with_cls

# name -> (dim, depth, heads, patch). mlp_ratio 4 everywhere, as in the ViT / DeiT papers.
VARIANTS = {
    "vit_ti16": (192, 12, 3, 16),     # DeiT-Ti / ViT-Ti      5.5M   fastest, good first run
    "vit_s16": (384, 12, 6, 16),      # ViT-S/16 (DeiT-S)    21.7M   MAIN MODEL
    "vit_s32": (384, 12, 6, 32),      # ViT-S/32             22.5M   49 tokens, ~4x cheaper than S/16
    "vit_m16": (512, 12, 8, 16),      # ViT-M/16 (mid)       38.3M   between S and B
    "vit_b16": (768, 12, 12, 16),     # ViT-B/16             85.8M   fits 12 GB with bf16 + small batch
    "vit_b32": (768, 12, 12, 32),     # ViT-B/32             87.5M   cheap B-width model
}


@dataclass
class ViTConfig:
    variant: str = "vit_s16"
    image_size: int = 224
    patch_size: int = 16
    dim: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    stem: str = "patch"              # patch (standard ViT) | conv (Xiao et al. 2021, "early convolutions")
    pool: str = "cls"                # cls (standard) | gap (mean of patch tokens)
    dropout: float = 0.0             # MLP / projection dropout
    attn_dropout: float = 0.0
    drop_path: float = 0.1           # stochastic depth, linearly increasing over depth (DeiT: 0.1 for S)
    layer_scale: float = 0.0         # >0 enables LayerScale (CaiT) with this init, e.g. 1e-4; 0 = off
    num_outputs: int = 1             # 1 logit, BCE (Cracked = 1)
    arch: str = "vit"
    attention: str = "mhsa"          # mhsa | str1 (skeleton-graph sparse attention)
    str_k: int = 8                   # STR-1: neighbours per token
    str_radius: float = 4.0          #        candidate radius (token cells)
    str_alpha: float = 30.0          #        tangent-cone half-angle (deg)
    str_beta: float = 30.0           #        max orientation difference (deg)
    str_rho: float = 1.0             #        distance scale in the edge score
    str_mode: str = "soft"           #        hard (mask only) | soft (mask + learned edge-score bias)
    str_dense_heads: int = 1         #        dense fallback heads per block
    str_coh_gate: bool = True        #        coherence gating of the orientation prior
    str_prior: str = "ridge"         #        ridge (Hessian dark-line) | gradient (plain structure tensor)
    str_sigma: float = 2.0           #        ridge scale in pixels

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_variant(cls, variant="vit_s16", **overrides):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {', '.join(VARIANTS)}")
        dim, depth, heads, patch = VARIANTS[variant]
        base = dict(variant=variant, dim=dim, depth=depth, num_heads=heads, patch_size=patch)
        base.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**base)


# ------------------------------------------------------------------ layers
class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class LayerScale(nn.Module):
    def __init__(self, dim, init):
        super().__init__()
        self.gamma = nn.Parameter(init * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class PatchStem(nn.Module):
    """Standard ViT patchify: one p x p conv with stride p."""

    def __init__(self, patch, dim):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, patch, patch)

    def forward(self, x):
        return self.proj(x)


class ConvStem(nn.Module):
    """ViT_C stem (Xiao et al., NeurIPS 2021): stride-2 3x3 convs (BN + ReLU) until the patch stride is
    reached, then a 1x1 conv to `dim`. Same token grid as the patch stem; usually trains more stably from
    scratch on small data. Only for power-of-two patch sizes."""

    def __init__(self, patch, dim):
        super().__init__()
        n = int(math.log2(patch))
        if 2 ** n != patch:
            raise ValueError("conv stem needs a power-of-two patch size")
        chans = [3] + [min(dim, dim // 2 ** (n - 1 - i)) for i in range(n)]
        layers = []
        for i in range(n):
            layers += [nn.Conv2d(chans[i], chans[i + 1], 3, 2, 1, bias=False), nn.BatchNorm2d(chans[i + 1]),
                       nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(chans[-1], dim, 1))
        self.proj = nn.Sequential(*layers)

    def forward(self, x):
        return self.proj(x)


def make_attention(c, rel_grid=None):
    return GeoAttention(c.dim, c.num_heads, c.qkv_bias, c.attn_dropout, c.dropout, c.attention, c.str_dense_heads,
                        c.str_mode == "soft", rel_grid)


class Mlp(nn.Module):
    def __init__(self, dim, hidden, drop=0.0):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, hidden), nn.Linear(hidden, dim)
        self.act, self.drop = nn.GELU(), nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block(nn.Module):
    def __init__(self, c, drop_path):
        super().__init__()
        dim, drop, ls_init = c.dim, c.dropout, c.layer_scale
        self.norm1, self.norm2 = nn.LayerNorm(dim, eps=1e-6), nn.LayerNorm(dim, eps=1e-6)
        self.attn = make_attention(c)
        self.mlp = Mlp(dim, int(dim * c.mlp_ratio), drop)
        self.ls1 = LayerScale(dim, ls_init) if ls_init > 0 else nn.Identity()
        self.ls2 = LayerScale(dim, ls_init) if ls_init > 0 else nn.Identity()
        self.drop_path = DropPath(drop_path)

    def forward(self, x, geo=None):
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x), geo)))
        return x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))


# ------------------------------------------------------------------ model
class ViT(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = c = cfg
        if c.image_size % c.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if c.dim % c.num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.grid = c.image_size // c.patch_size
        self.stem = (ConvStem if c.stem == "conv" else PatchStem)(c.patch_size, c.dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, c.dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.grid ** 2 + 1, c.dim))
        self.pos_drop = nn.Dropout(c.dropout)
        dpr = [c.drop_path * i / max(1, c.depth - 1) for i in range(c.depth)]
        self.blocks = nn.ModuleList(Block(c, dpr[i]) for i in range(c.depth))
        self.graph = (SkeletonGraph(self.grid, c.str_k, c.str_radius, c.str_alpha, c.str_beta, c.str_rho, c.str_mode,
                                    c.str_coh_gate, c.str_prior, c.str_sigma) if c.attention == "str1" else None)
        self.norm = nn.LayerNorm(c.dim, eps=1e-6)
        self.head = nn.Linear(c.dim, c.num_outputs)
        self._init()

    def _init(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                if isinstance(self.stem, PatchStem) and m is self.stem.proj:   # patch embed: like a Linear (timm)
                    nn.init.trunc_normal_(m.weight.view(m.weight.shape[0], -1), std=0.02)
                else:
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def tokens(self, x):
        x = self.stem(x).flatten(2).transpose(1, 2)                         # (B, grid^2, dim)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], 1)
        return self.pos_drop(x + self.pos_embed)

    def geometry(self, x):
        """STR-1 graph of the input images (None for MHSA)."""
        return with_cls(self.graph(x)) if self.graph is not None else None

    def features(self, x):
        geo = self.geometry(x)
        x = self.tokens(x)
        for blk in self.blocks:
            x = blk(x, geo)
        x = self.norm(x)
        return x[:, 0] if self.cfg.pool == "cls" else x[:, 1:].mean(1)

    def forward(self, x):
        """-> (B,) logits of Cracked (or (B, K) when num_outputs > 1)."""
        out = self.head(self.features(x))
        return out.squeeze(-1) if self.cfg.num_outputs == 1 else out

    # ---------------------------------------------------------------- reporting
    @torch.no_grad()
    def param_report(self):
        count = lambda m: sum(p.numel() for p in (m.parameters() if isinstance(m, nn.Module) else [m]))
        rows = {"stem": count(self.stem), "cls+pos_embed": count(self.cls_token) + count(self.pos_embed)}
        for i, b in enumerate(self.blocks):
            rows[f"block.{i}"] = count(b)
        rows["norm+head"] = count(self.norm) + count(self.head)
        return dict(rows=rows, total_params=sum(p.numel() for p in self.parameters()),
                    tokens=self.grid ** 2 + 1)

    def summary(self):
        c, r = self.cfg, self.param_report()
        line = "-" * 62
        out = [line, f"{c.variant} | dim {c.dim} | depth {c.depth} | heads {c.num_heads} | mlp {c.mlp_ratio:g}x | "
                     f"patch {c.patch_size} ({c.stem} stem) | {c.image_size}px",
               f"tokens {r['tokens']} ({self.grid}x{self.grid} + CLS) | pool {c.pool} | drop_path {c.drop_path} | "
               f"layer_scale {c.layer_scale or 'off'}", f"attention {attention_desc(c)}", line]
        rows = r["rows"]
        blk = [k for k in rows if k.startswith("block.")]
        out.append(f"  {'stem':<28}{rows['stem']:>16,}")
        out.append(f"  {'cls+pos_embed':<28}{rows['cls+pos_embed']:>16,}")
        out.append(f"  {f'blocks x{len(blk)} (each)':<28}{rows[blk[0]]:>16,}")
        out.append(f"  {'norm+head (1 logit)':<28}{rows['norm+head']:>16,}")
        out += [line, f"  {'TOTAL parameters':<28}{r['total_params']:>16,}  ({r['total_params'] / 1e6:.2f}M)", line]
        return "\n".join(out)


def attention_desc(c):
    if c.attention != "str1":
        return "MHSA (dense)"
    return (f"STR-1 skeleton-graph ({c.str_mode}, k={c.str_k}, radius={c.str_radius:g}, alpha={c.str_alpha:g}, "
            f"beta={c.str_beta:g}, {c.str_prior} prior, gate {'on' if c.str_coh_gate else 'off'}, "
            f"{c.str_dense_heads}/{c.num_heads} dense heads)")


def build_vit(variant="vit_s16", **overrides) -> ViT:
    return ViT(ViTConfig.from_variant(variant, **overrides))


if __name__ == "__main__":       # python vit.py  -> parameter table for every variant
    for v in VARIANTS:
        m = build_vit(v)
        print(f"{v:<10} {m.param_report()['total_params'] / 1e6:6.2f}M params | tokens {m.param_report()['tokens']}")
