"""
Paper outputs for one SDNET2018 run: a short metrics table and four figures, all derived from ONE
validation-tuned threshold and ONE test confusion matrix, so every number agrees with the others.
Borrowed from the LoopCrackViT repo (block applications replaced by GMACs).

Written automatically at the end of train.py and by evaluate.py (for any saved checkpoint), into
<run>/paper/ (or <run>/paper_<ckpt>/ for a checkpoint other than best.pt):
    metrics.md / .csv / .json   accuracy, macro P/R/F1, ROC-AUC, PR-AUC, crack recall/precision,
                                specificity, MCC, Brier, ECE, confusion counts, params, GMACs
    fig_training_curves.png     loss and validation ROC-AUC / MCC per epoch
    fig_roc_pr.png              ROC and precision-recall curves with the operating point
    fig_reliability.png         reliability diagram (15 bins) with Brier and ECE
    fig_confusion.png           test confusion matrix at the operating point

Mean +- s.d. over seeds (the paper table):
    python paper.py runs/<run_s42> runs/<run_s43> runs/<run_s44> --out runs/paper_mean_std.md
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

import metrics as M

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})

# (label, key) in table order
ROWS = [("Accuracy", "acc"), ("Precision (macro)", "macro_precision"), ("Recall (macro)", "macro_recall"),
        ("F1-score (macro)", "macro_f1"), ("ROC-AUC", "auc"), ("PR-AUC (average precision)", "ap"),
        ("Crack recall (sensitivity)", "recall"), ("Crack precision", "precision"), ("Specificity", "specificity"),
        ("MCC", "mcc"), ("Brier score", "brier"), ("Expected calibration error", "ece")]


def paper_metrics(y, p, thr):
    """Every reported metric from one confusion matrix at `thr`, plus the threshold-free ones."""
    m = M.compute_all(y, p, thr)
    tn, fp, fn, tp = confusion_matrix(y, (p >= thr).astype(int), labels=[0, 1]).ravel()
    d = lambda a, b: a / b if b else 0.0
    p1, r1, p0, r0 = d(tp, tp + fp), d(tp, tp + fn), d(tn, tn + fn), d(tn, tn + fp)
    f = lambda a, b: d(2 * a * b, a + b)
    out = {k: float(m[k]) for _, k in ROWS if k in m}
    out.update(acc=d(tp + tn, tp + tn + fp + fn), macro_precision=(p1 + p0) / 2, macro_recall=(r1 + r0) / 2,
               macro_f1=(f(p1, r1) + f(p0, r0)) / 2, recall=r1, precision=p1, specificity=r0,
               threshold=float(thr), tp=int(tp), fn=int(fn), fp=int(fp), tn=int(tn))
    return out


def plot_training(history, path):
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(h.epoch, h.train_loss, color=BLUE, lw=2, label="train")
    ax[0].plot(h.epoch, h.val_loss, color=ORANGE, lw=2, label="validation")
    ax[0].set(xlabel="Epoch", ylabel="Loss", title="Loss")
    ax[0].legend()
    ax[1].plot(h.epoch, h.val_auc, color=BLUE, lw=2, label="ROC-AUC")
    ax[1].plot(h.epoch, h.val_mcc, color=AQUA, lw=2, label="MCC")
    best = h.val_auc.idxmax()
    ax[1].scatter([h.epoch[best]], [h.val_auc[best]], s=40, color=BLUE, edgecolor="white", zorder=3)
    ax[1].annotate(f"best epoch {int(h.epoch[best])}", (h.epoch[best], h.val_auc[best]),
                   textcoords="offset points", xytext=(0, -16), ha="center", color=INK2, fontsize=9)
    ax[1].set(xlabel="Epoch", ylabel="Score", title="Validation ROC-AUC and MCC")
    ax[1].legend(loc="lower right")
    fig.savefig(path); plt.close(fig)


def plot_roc_pr(y, p, m, path):
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    ax[0].plot(fpr, tpr, color=BLUE, lw=2, label=f"ROC-AUC {m['auc']:.4f}")
    ax[0].plot([0, 1], [0, 1], color=INK2, lw=1, ls="--", label="chance")
    ax[0].scatter([1 - m["specificity"]], [m["recall"]], s=50, color=ORANGE, edgecolor="white", zorder=3,
                  label=f"operating point (t = {m['threshold']:.2f})")
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate (crack recall)", title="ROC curve",
              xlim=(0, 1), ylim=(0, 1.01))
    ax[0].legend(loc="lower right")
    ax[1].plot(rec, prec, color=BLUE, lw=2, label=f"PR-AUC {m['ap']:.4f}")
    ax[1].axhline(y.mean(), color=INK2, lw=1, ls="--", label=f"prevalence {y.mean():.3f}")
    ax[1].scatter([m["recall"]], [m["precision"]], s=50, color=ORANGE, edgecolor="white", zorder=3,
                  label=f"operating point (t = {m['threshold']:.2f})")
    ax[1].set(xlabel="Crack recall", ylabel="Crack precision", title="Precision-recall curve",
              xlim=(0, 1), ylim=(0, 1.01))
    ax[1].legend(loc="center left")
    fig.savefig(path); plt.close(fig)


def plot_reliability(y, p, m, path, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    conf = np.array([p[idx == b].mean() if (idx == b).any() else np.nan for b in range(bins)])
    freq = np.array([y[idx == b].mean() if (idx == b).any() else np.nan for b in range(bins)])
    count = np.bincount(idx, minlength=bins)
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--", label="perfect calibration")
    ok = count > 0
    ax.plot(conf[ok], freq[ok], "o-", color=BLUE, lw=2, ms=6, mec="white", label="model")
    ax.set(xlabel="Predicted crack probability", ylabel="Observed crack frequency", title="Reliability diagram",
           xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left")
    ax.text(0.97, 0.04, f"Brier {m['brier']:.4f}\nECE {m['ece']:.4f}", transform=ax.transAxes,
            ha="right", va="bottom", color=INK)
    fig.savefig(path); plt.close(fig)


def plot_confusion(m, path):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm / np.maximum(cm.sum(1, keepdims=True), 1), cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            frac = cm[i, j] / max(cm[i].sum(), 1)
            ax.text(j, i, f"{cm[i, j]:,}\n({frac:.1%})", ha="center", va="center",
                    color="white" if frac > 0.5 else INK)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Non-cracked", "Cracked"],
           yticklabels=["Non-cracked", "Cracked"], xlabel="Predicted", ylabel="True",
           title=f"Test confusion matrix (t = {m['threshold']:.2f})")
    ax.grid(False)
    fig.savefig(path); plt.close(fig)


def write_report(out, run_name, yv, pv, yt, pt, n_params, gmacs=None, history=None, ckpt="best.pt", epoch=None):
    """yv/pv, yt/pt: labels and P(Cracked) on VAL and TEST. Threshold = val-best F1."""
    d = os.path.join(out, "paper" if ckpt == "best.pt" else f"paper_{os.path.splitext(ckpt)[0]}")
    os.makedirs(d, exist_ok=True)
    yv, yt = np.asarray(yv).astype(int), np.asarray(yt).astype(int)
    pv, pt = np.asarray(pv, float), np.asarray(pt, float)
    thr, _ = M.best_f1_threshold(yv, pv)
    m = paper_metrics(yt, pt, thr)

    rows = [(label, f"{m[k]:.4f}") for label, k in ROWS]
    rows += [("Confusion (TP / FN / FP / TN)", f"{m['tp']} / {m['fn']} / {m['fp']} / {m['tn']}"),
             ("Threshold (val-best F1)", f"{thr:.3f}"), ("Parameters (M)", f"{n_params / 1e6:.2f}")]
    if gmacs:
        rows.append(("GMACs (1 image)", f"{gmacs:.2f}"))
    tab = pd.DataFrame(rows, columns=["Metric", "Value"])
    tab.to_csv(os.path.join(d, "metrics.csv"), index=False)
    md = [f"# {run_name}", "",
          f"Checkpoint `{ckpt}`" + (f" (epoch {epoch})" if epoch else "") +
          f". Test N = {len(yt):,} ({int(yt.sum()):,} cracked, prevalence {yt.mean():.3f}). "
          f"All thresholded metrics use one threshold chosen on VAL (best F1) and one test confusion matrix.",
          "", tab.to_markdown(index=False), ""]
    with open(os.path.join(d, "metrics.md"), "w", encoding="utf8") as f:
        f.write("\n".join(md))
    with open(os.path.join(d, "metrics.json"), "w") as f:
        json.dump(dict(run=run_name, ckpt=ckpt, epoch=epoch, params=int(n_params), gmacs=gmacs, **m), f, indent=2)

    if history:
        plot_training(history, os.path.join(d, "fig_training_curves.png"))
    plot_roc_pr(yt, pt, m, os.path.join(d, "fig_roc_pr.png"))
    plot_reliability(yt, pt, m, os.path.join(d, "fig_reliability.png"))
    plot_confusion(m, os.path.join(d, "fig_confusion.png"))
    print("\n" + "\n".join(md[2:]) + f"\npaper outputs -> {d}")
    return m


def mean_std(runs, out, sub="paper"):
    recs = [json.load(open(os.path.join(r, sub, "metrics.json"))) for r in runs]
    df = pd.DataFrame(recs)
    rows = [(label, f"{df[k].mean():.4f} ± {df[k].std(ddof=1) if len(df) > 1 else 0:.4f}") for label, k in ROWS]
    rows.append(("Parameters (M)", f"{df.params.iloc[0] / 1e6:.2f}"))
    tab = pd.DataFrame(rows, columns=["Metric", f"mean ± s.d. (n = {len(df)})"])
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf8") as f:
        f.write("Runs: " + ", ".join(df.run) + "\n\n" + tab.to_markdown(index=False) + "\n")
    tab.to_csv(os.path.splitext(out)[0] + ".csv", index=False)
    print(tab.to_string(index=False) + f"\n-> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories that already have paper/metrics.json")
    ap.add_argument("--sub", default="paper", help="paper (best.pt) or paper_last, paper_epoch_0040 ...")
    ap.add_argument("--out", default="runs/paper_mean_std.md")
    a = ap.parse_args()
    mean_std(a.runs, a.out, a.sub)
