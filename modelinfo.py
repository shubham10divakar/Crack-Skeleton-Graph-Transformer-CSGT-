"""
Full model report, printed at the start of every training run (and by --summary-only), saved to <run>/model.txt:

    1. compact summary (architecture, attention, parameter groups)
    2. full model config
    3. layer table: every module, indented by depth, with type, output shape (one dummy forward pass),
       own parameters and parameters including children
    4. totals: parameters (trainable / frozen), weights size, full-checkpoint size estimate, GMACs

    python modelinfo.py --variant vit_s16 --stem conv --attention str1      (same flags as train.py)
"""
from __future__ import annotations

import json

import torch
import torch.nn as nn


def _shape(o):
    if isinstance(o, torch.Tensor):
        return "x".join(str(s) for s in o.shape)
    if isinstance(o, (tuple, list)) and o and isinstance(o[0], torch.Tensor):
        return ", ".join(_shape(t) for t in o)
    if isinstance(o, dict):
        return "dict(" + ", ".join(o)[:14] + ")"
    return ""


@torch.no_grad()
def layer_table(model, image_size, device):
    """-> list of (name, depth, type, output shape, own params, total params), parents before children."""
    shapes, hooks = {}, []
    for name, m in model.named_modules():
        if name == "":
            continue

        def hook(mod, inp, out, name=name):
            shapes[name] = _shape(out)
        hooks.append(m.register_forward_hook(hook))
    was_training = model.training
    model.eval()
    try:
        model(torch.zeros(1, 3, image_size, image_size, device=device))
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    mods = dict(model.named_modules())
    rows = []
    for name in [n for n in mods if n]:
        m = mods[name]
        own = sum(p.numel() for p in m.parameters(recurse=False))
        tot = sum(p.numel() for p in m.parameters())
        rows.append((name, name.count("."), type(m).__name__, shapes.get(name, "-"), own, tot))
    return rows


def gmacs(model, image_size, device):
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            model(torch.zeros(1, 3, image_size, image_size, device=device))
        return fc.get_total_flops() / 2e9
    except Exception:
        return None


def model_report(model, cfg, device):
    """Full printable report (str)."""
    size = cfg.image_size
    lines = [model.summary(), "", "MODEL CONFIG", json.dumps(cfg.to_dict(), indent=1, default=str), "",
             "LAYERS (module tree; output shape for one 3x{0}x{0} image; '-' = not called directly)".format(size)]
    rows = layer_table(model, size, device)
    w = max(len("  " * d + n.split(".")[-1] if d else n) for n, d, *_ in rows) + 2
    w = min(max(w, 30), 60)
    head = f"{'layer':<{w}}{'type':<22}{'output shape':<22}{'params (own)':>14}{'params (total)':>16}"
    lines += [head, "-" * len(head)]
    for name, d, typ, shp, own, tot in rows:
        label = ("  " * d + name.split(".")[-1]) if d else name
        lines.append(f"{label[:w - 1]:<{w}}{typ[:21]:<22}{shp[:21]:<22}{own:>14,}{tot:>16,}")
    n_all = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_buf = sum(b.numel() for b in model.buffers())
    g = gmacs(model, size, device)
    mb = n_all * 4 / 2 ** 20
    lines += ["-" * len(head),
              f"TOTAL parameters        {n_all:,}  ({n_all / 1e6:.2f}M)",
              f"  trainable             {n_tr:,}",
              f"  frozen                {n_all - n_tr:,}",
              f"buffers                 {n_buf:,}",
              f"weights size (fp32)     {mb:.1f} MB   | full checkpoint with AdamW state ~{3 * mb:.0f} MB",
              f"compute                 {g:.2f} GMACs per {size}x{size} image (forward)" if g else "compute  (FLOP count unavailable)"]
    return "\n".join(lines)


if __name__ == "__main__":
    from train import get_args
    from models import config_from_args, model_from_cfg
    a = get_args()
    c = config_from_args(a)
    print(model_report(model_from_cfg(c.to_dict()), c, torch.device("cpu")))
