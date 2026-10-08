"""
Model factory: one `--variant` flag selects the architecture.

    vit_ti16 | vit_s16 | vit_s32 | vit_m16 | vit_b16 | vit_b32     plain ViT (vit.py), --attention mhsa | str1
    hyb_r26  | hyb_r50                                             ResNet-style conv + attention (hybrid.py)

Checkpoints store the config dict; `model_from_cfg` rebuilds the right class (configs saved before the
hybrid existed have no "arch" key and are ViTs).
"""
from __future__ import annotations

import torch

from hybrid import HYBRID_VARIANTS, HybridConfig, HybridNet
from vit import VARIANTS, ViT, ViTConfig

ALL_VARIANTS = list(VARIANTS) + list(HYBRID_VARIANTS)
# --task six (CrackNeXt protocol): class index = 2 * surface + cracked
SIX_CLASSES = [f"{s}_{c}" for s in ("Decks", "Pavements", "Walls") for c in ("Non-cracked", "Cracked")]


def crack_logit(out):
    """Model output -> Cracked logit. A 6-class output collapses to log P(cracked) - log P(non-cracked) over the
    surfaces, so sigmoid() of it is exactly the summed softmax of the three Cracked classes; every binary metric,
    threshold and Grad-CAM therefore works unchanged for both tasks."""
    if out.ndim == 1:
        return out
    out = out.float()
    return torch.logsumexp(out[:, 1::2], 1) - torch.logsumexp(out[:, 0::2], 1)


def config_from_dict(d):
    return (HybridConfig if d.get("arch", "vit") == "hybrid" else ViTConfig)(**d)


def normalized(d):
    """Config dict with defaults filled in (so configs from older code compare equal)."""
    return config_from_dict(d).to_dict()


def model_from_cfg(d):
    cfg = config_from_dict(d)
    return HybridNet(cfg) if cfg.arch == "hybrid" else ViT(cfg)


def config_from_args(a):
    common = dict(image_size=a.image_size, attention=a.attention, str_k=a.str_k, str_radius=a.str_radius,
                  str_alpha=a.str_alpha, str_beta=a.str_beta, str_rho=a.str_rho, str_mode=a.str_mode,
                  str_dense_heads=a.str_dense_heads, str_coh_gate=a.str_coh_gate, str_prior=a.str_prior,
                  str_sigma=a.str_sigma, drop_path=a.drop_path, num_outputs=6 if a.task == "six" else 1)
    if a.variant in HYBRID_VARIANTS:
        stages = "" if a.hybrid_attn_stages is None else a.hybrid_attn_stages   # --hybrid-attn-stages none
        return HybridConfig.from_variant(a.variant, attn_stages=stages, **common)
    if a.variant not in VARIANTS:
        raise SystemExit(f"--variant must be one of {', '.join(ALL_VARIANTS)}")
    return ViTConfig.from_variant(
        a.variant, patch_size=a.patch_size, dim=a.dim, depth=a.depth, num_heads=a.num_heads, mlp_ratio=a.mlp_ratio,
        qkv_bias=a.qkv_bias, stem=a.stem, pool=a.pool, dropout=a.dropout, attn_dropout=a.attn_dropout,
        layer_scale=a.layer_scale, **common)


def arch_tag(c):
    """Run-name fragment describing the architecture and attention."""
    if c.arch == "hybrid":
        tag = f"{c.variant}-a{c.attn_stages.replace(',', '') or '0'}"
        if not c.attn_stages:
            return tag + ("_6c" if c.num_outputs == 6 else "")
    else:
        d = ViTConfig.from_variant(c.variant)
        tag = c.variant
        if (c.dim, c.depth, c.num_heads, c.patch_size) != (d.dim, d.depth, d.num_heads, d.patch_size):
            tag += f"-d{c.dim}x{c.depth}h{c.num_heads}p{c.patch_size}"
        tag += ("_conv" if c.stem == "conv" else "") + ("_gap" if c.pool == "gap" else "")
    tag += f"_r{c.image_size}" if c.image_size != 224 else ""
    tag += "_6c" if c.num_outputs == 6 else ""
    if c.attention == "str1":
        s = HybridConfig()                                   # defaults (same for both architectures)
        tag += f"_str1-{c.str_mode}-k{c.str_k}-d{c.str_dense_heads}"
        extra = [f"r{c.str_radius:g}" if c.str_radius != s.str_radius else "",
                 f"a{c.str_alpha:g}" if c.str_alpha != s.str_alpha else "",
                 f"b{c.str_beta:g}" if c.str_beta != s.str_beta else "",
                 "nocoh" if not c.str_coh_gate else "",
                 "grad" if c.str_prior == "gradient" else "",
                 f"sig{c.str_sigma:g}" if c.str_sigma != s.str_sigma else ""]
        tag += "".join("-" + e for e in extra if e)
    elif c.arch == "hybrid":
        tag += "_mhsa"
    return tag
