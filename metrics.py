"""
Metrics for SDNET2018 crack classification (positive class = Cracked). Borrowed from the LoopCrackViT repo.

On the imbalanced splits (random / group, ~15% cracked) accuracy is nearly meaningless, so the headline
numbers are the threshold-free ranking metrics (ROC-AUC, PR-AUC/AP) plus the crack-class operating-point
metrics a structural inspector cares about: recall (missed cracks = FN), precision, F1/F2, specificity,
MCC, IoU, and calibration (Brier, ECE).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, cohen_kappa_score,
                             confusion_matrix, log_loss, matthews_corrcoef,
                             precision_recall_curve, roc_auc_score)


def ece_score(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def _f1_neg(tn, fp, fn):
    p = tn / (tn + fn) if (tn + fn) else 0.0
    r = tn / (tn + fp) if (tn + fp) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def compute_all(y, p, thr=0.5):
    """y: {0,1} (1 = Cracked); p: P(Cracked). Returns a flat dict of floats."""
    y, p = np.asarray(y).astype(int), np.asarray(p).astype(float)
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    d = lambda a, b: float(a / b) if b else 0.0
    prec, rec, spec = d(tp, tp + fp), d(tp, tp + fn), d(tn, tn + fp)
    fb = lambda beta: d((1 + beta ** 2) * prec * rec, beta ** 2 * prec + rec)
    two = len(np.unique(y)) > 1
    return dict(
        threshold=float(thr), n=int(len(y)), tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
        acc=d(tp + tn, len(y)), bal_acc=0.5 * (rec + spec),
        precision=prec, recall=rec, specificity=spec, npv=d(tn, tn + fn),
        f1=fb(1), f2=fb(2), f1_macro=0.5 * (fb(1) + _f1_neg(tn, fp, fn)),
        mcc=float(matthews_corrcoef(y, pred)) if two else 0.0,
        kappa=float(cohen_kappa_score(y, pred)) if two else 0.0,
        iou_crack=d(tp, tp + fp + fn),
        fpr=d(fp, fp + tn), fnr=d(fn, fn + tp),
        auc=float(roc_auc_score(y, p)) if two else float("nan"),
        ap=float(average_precision_score(y, p)) if two else float("nan"),
        brier=float(brier_score_loss(y, p)), ece=ece_score(y, p),
        nll=float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
    )


def best_f1_threshold(y, p, beta=1.0):
    """Threshold maximising F-beta on (y, p). Tune on VAL, then apply to TEST."""
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    f = (1 + beta ** 2) * prec * rec / (beta ** 2 * prec + rec + 1e-12)
    i = int(np.argmax(f))
    return float(thr[i]), float(f[i])


def threshold_at_recall(y, p, target=0.95):
    """Highest threshold that still reaches `target` recall (fewest false alarms)."""
    prec, rec, thr = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= target)[0]
    return float(thr[ok[-1]]) if len(ok) else float(thr[0])


def bootstrap_ci(y, p, thr, keys=("auc", "ap", "f1", "recall", "precision", "mcc"),
                 n_boot=1000, seed=0, alpha=0.05):
    """95% CIs from a stratified bootstrap of the test set."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    acc = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        m = compute_all(y[idx], p[idx], thr)
        for k in keys:
            acc[k].append(m[k])
    return {k: (float(np.percentile(v, 100 * alpha / 2)), float(np.percentile(v, 100 * (1 - alpha / 2))))
            for k, v in acc.items()}
