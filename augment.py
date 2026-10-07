"""
Batch-level augmentation for the SDNET (binary) trainer: MixUp, CutMix, and SMOTE-style minority synthesis.

All functions run on GPU tensors inside the training loop, after the dataloader, and only during training.
Targets are float tensors in [0, 1] (1 = Cracked), so mixed samples get soft labels.

  mix_batch     MixUp:  x = lam*xa + (1-lam)*xb,  y = lam*ya + (1-lam)*yb
                CutMix: paste a random box from xb into xa, y weighted by the box AREA.
                Caveat for cracks: the label follows box area, but a hairline crack covers few pixels, so a
                pasted box can delete the crack while the label still says "mostly cracked". Treat it as a
                regulariser and check it on the validation split (it is an ablation flag, off by default).
  smote_batch   SMOTE-style oversampling. Classic SMOTE interpolates between a minority sample and one of its
                nearest neighbours in feature space; on raw pixels that is impractical, so this interpolates
                between TWO random cracked images from a rolling bank of recent ones (a MixUp within the
                minority class, hard label 1) and appends the synthetic images to the batch until the
                cracked fraction reaches `target`.
"""
from __future__ import annotations

import math

import numpy as np
import torch


def mix_batch(x, y, mixup_alpha=0.0, cutmix_alpha=0.0, prob=1.0, switch_prob=0.5):
    """MixUp and/or CutMix on a batch. x: (B,C,H,W); y: (B,) or (B,K) float in [0,1]. Returns (x, y)."""
    if (mixup_alpha <= 0 and cutmix_alpha <= 0) or x.shape[0] < 2 or np.random.rand() > prob:
        return x, y
    use_cut = cutmix_alpha > 0 and (mixup_alpha <= 0 or np.random.rand() < switch_prob)
    alpha = cutmix_alpha if use_cut else mixup_alpha
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.shape[0], device=x.device)
    if use_cut:
        H, W = x.shape[-2:]
        r = math.sqrt(1.0 - lam)
        ch, cw = int(H * r), int(W * r)
        cy, cx = np.random.randint(H), np.random.randint(W)
        y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, H)
        x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, W)
        x = x.clone()
        x[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
        lam = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)
    else:
        x = lam * x + (1.0 - lam) * x[perm]
    return x, lam * y + (1.0 - lam) * y[perm]


class MinorityBank:
    """Rolling store of recent minority-class (cracked) images, kept in half precision on the GPU."""

    def __init__(self, capacity=256):
        self.capacity, self.buf = capacity, None

    def __len__(self):
        return 0 if self.buf is None else self.buf.shape[0]

    def add(self, xs):
        if xs.shape[0] == 0:
            return
        xs = xs.detach().half()
        self.buf = xs if self.buf is None else torch.cat([self.buf, xs])[-self.capacity:]

    def synthesize(self, n):
        """n images, each a random convex combination of two different bank images (None if < 2 stored)."""
        m = len(self)
        if m < 2:
            return None
        i = torch.randint(0, m, (n,), device=self.buf.device)
        j = (i + torch.randint(1, m, (n,), device=self.buf.device)) % m
        lam = torch.rand(n, 1, 1, 1, device=self.buf.device)
        return lam * self.buf[i].float() + (1.0 - lam) * self.buf[j].float()


def smote_batch(x, y, bank, target=0.5, max_extra_frac=0.5):
    """Append synthetic cracked images so that roughly `target` of the batch is cracked.
    Adds at most `max_extra_frac * B` images (bounds the memory increase). Returns (x, y)."""
    pos = y >= 0.5
    bank.add(x[pos])
    B, n_min = y.shape[0], int(pos.sum())
    extra = int(round((target * B - n_min) / (1.0 - target)))
    extra = min(max(extra, 0), int(B * max_extra_frac))
    if extra == 0:
        return x, y
    xs = bank.synthesize(extra)
    if xs is None:
        return x, y
    return torch.cat([x, xs.to(x.dtype)]), torch.cat([y, torch.ones(extra, device=y.device, dtype=y.dtype)])
