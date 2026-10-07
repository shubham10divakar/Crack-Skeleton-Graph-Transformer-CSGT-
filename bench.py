"""
Measure peak GPU memory and training speed of each ViT variant on THIS machine (synthetic data, a few
bf16 training steps), to choose --variant / --batch-size / --grad-accum. Run it while the GPU is idle.

    python bench.py                                   # all variants, batch 64 and 32
    python bench.py --variants vit_s16 vit_b16 --batch-sizes 128 64 32
    python bench.py --variants vit_s16 hyb_r26 hyb_r50 --stem conv --attention str1
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from hybrid import HYBRID_VARIANTS, HybridConfig, HybridNet
from vit import VARIANTS, build_vit


def build(variant, size, stem, attention):
    if variant in HYBRID_VARIANTS:
        return HybridNet(HybridConfig.from_variant(variant, image_size=size, attention=attention))
    return build_vit(variant, image_size=size, stem=stem, attention=attention)


def bench(variant, bs, size, steps, device, stem="patch", attention="mhsa"):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model = build(variant, size, stem, attention).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(bs, 3, size, size, device=device)
    y = torch.randint(0, 2, (bs,), device=device).float()
    try:
        for i in range(steps + 3):
            if i == 3:
                torch.cuda.synchronize(); t = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.binary_cross_entropy_with_logits(model(x).float(), y)
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return dict(img_s=bs * steps / (time.time() - t), peak_gb=torch.cuda.max_memory_allocated() / 2**30)
    except torch.OutOfMemoryError:
        return dict(img_s=float("nan"), peak_gb=float("inf"))
    finally:
        del model, opt, x
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 32])
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--stem", default="patch", help="ViT stem: patch | conv")
    ap.add_argument("--attention", default="mhsa", help="mhsa | str1")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("bench.py needs a CUDA GPU")
    dev = torch.device("cuda")
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    used = (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 2**30
    print(f"{torch.cuda.get_device_name(0)} | {total:.1f} GB total, {used:.1f} GB already in use by other processes"
          f" | stem={a.stem} attention={a.attention}")
    print(f"{'variant':<10}{'params':>9}{'batch':>7}{'peak GB':>10}{'train img/s':>13}{'~min/epoch (11,877 imgs)':>27}")
    for v in a.variants:
        n = sum(p.numel() for p in build(v, a.image_size, a.stem, a.attention).parameters()) / 1e6
        for bs in a.batch_sizes:
            r = bench(v, bs, a.image_size, a.steps, dev, a.stem, a.attention)
            ep = 11877 / r["img_s"] / 60 if r["img_s"] == r["img_s"] else float("nan")
            print(f"{v:<10}{n:>8.1f}M{bs:>7}{r['peak_gb']:>10.2f}{r['img_s']:>13.0f}{ep:>27.1f}"
                  + ("   OOM" if r["peak_gb"] == float("inf") else ""))


if __name__ == "__main__":
    main()
