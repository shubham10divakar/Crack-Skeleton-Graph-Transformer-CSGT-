"""
ImageNet-1k initialisation (--pretrained). Our own model classes are kept; matching weights are copied in from
the timm checkpoint below by name, so pretrained and from-scratch runs use exactly the same architecture and
differ only in the starting weights. Pretrained runs are written to runs/pretrained/ (train.py).

    hyb_r26  <- resnet26d.bt_in1k          ResNet-D, layout (2,2,2,2): our stage 3 has one extra block, which
                                           stays randomly initialised (zero-init last BN -> starts as identity)
    hyb_r50  <- resnet50d.ra2_in1k         ResNet-D (3,4,6,3): every conv block matches
    vit_ti16 <- deit_tiny_patch16_224.fb_in1k
    vit_s16  <- deit_small_patch16_224.fb_in1k
    vit_b16  <- deit_base_patch16_224.fb_in1k
    vit_b32  <- vit_base_patch32_224.augreg_in1k
    vit_s32  <- vit_small_patch32_224.augreg_in21k_ft_in1k   (no IN-1k-only weights exist; uses ImageNet-21k)
    vit_m16  -  no pretrained weights

Never pretrained (always random): the 1-logit head, the attention of hybrid attention blocks (they replace a
3x3 conv, as in BoTNet; their 1x1 reduce / expand convs are loaded but the block still starts as identity), the
STR-1 extras (edge-score gamma, relative position bias) and a ViT conv stem.
"""
from __future__ import annotations

import re

import torch
import torch.nn.functional as F

SOURCES = {"hyb_r26": "resnet26d.bt_in1k", "hyb_r50": "resnet50d.ra2_in1k",
           "vit_ti16": "deit_tiny_patch16_224.fb_in1k", "vit_s16": "deit_small_patch16_224.fb_in1k",
           "vit_b16": "deit_base_patch16_224.fb_in1k", "vit_b32": "vit_base_patch32_224.augreg_in1k",
           "vit_s32": "vit_small_patch32_224.augreg_in21k_ft_in1k"}

STEM = {"0": "stem.0.0", "1": "stem.0.1", "3": "stem.1.0", "4": "stem.1.1", "6": "stem.2.0"}
BLOCK = {"conv1": "reduce.0", "bn1": "reduce.1", "conv2": "conv.0", "bn2": "conv.1", "conv3": "expand.0",
         "bn3": "expand.1", "downsample.1": "short.1.0", "downsample.2": "short.1.1"}


def source_for(cfg, override=None):
    name = override or SOURCES.get(cfg.variant)
    if not name:
        raise SystemExit(f"--pretrained: no ImageNet weights for {cfg.variant}; pass --pretrained-model <timm name>")
    return name


def resnet_key(k):
    """timm ResNet-D key -> hybrid.py key."""
    if m := re.fullmatch(r"conv1\.(\d)\.(.+)", k):
        return f"{STEM[m[1]]}.{m[2]}"
    if k.startswith("bn1."):
        return "stem.2.1." + k[4:]
    if m := re.fullmatch(r"layer(\d)\.(\d+)\.(conv\d|bn\d|downsample\.\d)\.(.+)", k):
        return f"stages.{int(m[1]) - 1}.{m[2]}.{BLOCK[m[3]]}.{m[4]}"
    return None


def vit_key(k):
    return "stem." + k[len("patch_embed."):] if k.startswith("patch_embed.") else k


def resize_pos_embed(pe, n_tokens):
    """(1, 1 + g*g, D) -> (1, n_tokens, D), bicubic on the patch grid (CLS kept)."""
    cls, grid = pe[:, :1], pe[:, 1:]
    g, g2 = int(grid.shape[1] ** 0.5), int((n_tokens - 1) ** 0.5)
    grid = grid.reshape(1, g, g, -1).permute(0, 3, 1, 2)
    grid = F.interpolate(grid.float(), size=(g2, g2), mode="bicubic", align_corners=False)
    return torch.cat([cls, grid.permute(0, 2, 3, 1).reshape(1, g2 * g2, -1).to(pe.dtype)], 1)


def load_pretrained(model, cfg, override=None):
    """Copy ImageNet weights into `model` in place. -> printable report (source, coverage, what stays random)."""
    import timm
    name = source_for(cfg, override)
    src = timm.create_model(name, pretrained=True, num_classes=0).state_dict()
    own = model.state_dict()
    hybrid = cfg.arch == "hybrid"
    attn_blocks = {n[:-len(".attn")] for n, _ in model.named_modules() if n.endswith(".attn") and n.startswith("stages.")}
    load, unused = {}, []
    for k, v in src.items():
        t = resnet_key(k) if hybrid else vit_key(k)
        if t is None or t not in own:
            unused.append(k); continue
        if hybrid and t.rsplit(".expand.1.", 1)[0] in attn_blocks:   # attention block: keep zero-init last BN
            continue
        if t == "pos_embed" and v.shape != own[t].shape:
            v = resize_pos_embed(v, own[t].shape[1])
        if v.shape != own[t].shape:
            unused.append(f"{k} (shape {tuple(v.shape)} vs {tuple(own[t].shape)})"); continue
        load[t] = v
    if not load:
        raise SystemExit(f"--pretrained: no weights of {name} match {cfg.variant}")
    model.load_state_dict(load, strict=False)

    params = dict(model.named_parameters())
    n_loaded = sum(params[k].numel() for k in load if k in params)
    n_total = sum(p.numel() for p in params.values())
    unit = lambda k: ".".join(k.split(".")[:3 if k.startswith("stages.") else 2 if k.startswith("blocks.") else 1])
    groups = {}                                         # block / top module -> its randomly initialised params
    for k in params:
        groups.setdefault(unit(k), [[], []])[k in load].append(k[len(unit(k)) + 1:])
    rand = [u if not done else f"{u}[{', '.join(sorted({r.rsplit('.', 1)[0] if '.' in r else r for r in todo}))}]"
            for u, (todo, done) in groups.items() if todo]
    rand = sorted({re.sub(r"^blocks\.\d+", "blocks.*", r) for r in rand})   # same in every ViT block -> once
    lines = [f"pretrained: {name} (ImageNet) -> {len(load)} tensors, {n_loaded:,} of {n_total:,} parameters "
             f"({100 * n_loaded / n_total:.1f}%)",
             f"  random init: {', '.join(rand)}"]
    if unused:
        lines.append(f"  unused source tensors: {len(unused)} (e.g. {', '.join(unused[:3])})")
    return "\n".join(lines)
