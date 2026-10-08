"""
Evaluate a saved ViT run on the TEST split and write the paper metrics table.

    python evaluate.py --run runs/vit_s16_bs64_balanced_basic_s42                  # best.pt
    python evaluate.py --run runs/vit_s16_bs64_balanced_basic_s42 --ckpt last.pt   # any checkpoint
    python evaluate.py --run runs/vit_s16_bs64_balanced_basic_s42 --ckpt all       # every epoch_NNNN.pt

Use it when training ended before train.py's final evaluation, or to re-score another checkpoint.
Rebuilds the model from the checkpoint, uses the cached split, tunes the decision thresholds on VAL only
and applies them to TEST. EMA weights are used when the run trained with --ema-decay.

Writes to <run>/eval/ (best.pt) or <run>/eval_<ckpt>/:
    metrics_table.md / .csv   ROC-AUC, PR-AUC, crack recall/precision, specificity, MCC, Brier, ECE,
                              params, GMACs + 95% stratified-bootstrap CIs, at 0.5 and at the val-F1 threshold
    operating_points.csv      all metrics at thr 0.5 / val-best-F1 / val-recall target
    per_surface.csv           Decks / Pavements / Walls
    test_predictions.csv      P(Cracked) per test image
    metrics.json              everything above in one file
plus the paper table and figures (paper.py) in <run>/paper/ or <run>/paper_<ckpt>/.
--ckpt all writes <run>/eval_epochs.csv + fig_eval_epochs.png (val and test metrics of every saved epoch;
pick epochs by VAL only - choosing by test is test-set leakage).
Predictions are cached in <eval dir>/_preds.npz (pass --no-cache to recompute).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it, *a, **k):
        return it

import metrics as M
from data import CrackDataset, add_six_class, build_transforms, load_split
from paper import write_report
from models import SIX_CLASSES, crack_logit, model_from_cfg

TABLE = [("ROC-AUC", "auc", ".4f"), ("PR-AUC (average precision)", "ap", ".4f"), ("Accuracy", "acc", ".4f"),
         ("Crack recall (sensitivity)", "recall", ".4f"), ("Crack precision", "precision", ".4f"),
         ("Specificity", "specificity", ".4f"), ("F1 (Cracked)", "f1", ".4f"), ("MCC", "mcc", ".4f"),
         ("Brier score", "brier", ".4f"), ("Expected calibration error", "ece", ".4f")]
THRESHOLD_FREE = {"auc", "ap", "brier", "ece"}


def pick_device(name):
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else
                        "cpu" if name == "auto" else name)


def load_run(run, ckpt="best.pt", device="cpu"):
    """-> (model in eval mode with the evaluated weights (EMA if present), training args, checkpoint dict)."""
    cfg = json.load(open(os.path.join(run, "config.json")))
    ck = torch.load(os.path.join(run, ckpt), map_location="cpu", weights_only=False)
    model = model_from_cfg(ck.get("model_cfg", cfg["model"]))
    model.load_state_dict(ck["ema"] if ck.get("ema") is not None else ck["model"])
    return model.to(device).eval(), cfg["args"], ck


def run_split(targs, run):
    split_dir = targs.get("split_dir") or os.path.join(os.path.dirname(os.path.normpath(run)), "_splits")
    df = load_split(targs["data_root"], targs["split_mode"], targs["seed"], split_dir,
                    targs.get("val_frac", 0.15), targs.get("test_frac", 0.15))
    return (df[df["split"] == s].reset_index(drop=True) for s in ("val", "test"))


@torch.no_grad()
def predict(model, df, tf, device, bs, workers, amp, desc):
    """-> P(Cracked) (N,), and the 6-class softmax (N, 6) for --task six runs (else None)."""
    dl = DataLoader(CrackDataset(df, tf), batch_size=bs, shuffle=False, num_workers=workers,
                    pin_memory=device.type == "cuda")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    out = []
    for x, _ in tqdm(dl, desc=desc, dynamic_ncols=True, leave=False):
        with torch.autocast(device.type, dtype=dtype, enabled=amp and device.type == "cuda"):
            out.append(model(x.to(device, non_blocking=True)).float().cpu())
    lg = torch.cat(out)
    return torch.sigmoid(crack_logit(lg)).numpy(), (torch.softmax(lg, 1).numpy() if lg.ndim == 2 else None)


def gmacs(model, size, device):
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            model(torch.randn(1, 3, size, size, device=device))
        return fc.get_total_flops() / 2e9
    except Exception:
        return None


def get_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt", help="best.pt | last.pt | epoch_0040.pt | all")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--bootstrap", type=int, default=1000, help="stratified bootstrap resamples (0 = no CI)")
    ap.add_argument("--target-recall", type=float, default=None, help="default: value from the run config")
    ap.add_argument("--no-amp", action="store_true", help="fp32 inference (default: bf16 autocast like training)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--device", default="auto")
    return ap.parse_args()


def evaluate_one(args, run, ckpt_name, device):
    model, targs, ck = load_run(run, ckpt_name, device)
    stem = os.path.splitext(ckpt_name)[0]
    out = os.path.join(run, "eval" if ckpt_name == "best.pt" else f"eval_{stem}")
    os.makedirs(out, exist_ok=True)
    n_params = model.param_report()["total_params"]
    print(f">>> {os.path.basename(run)} | {ckpt_name} (epoch {ck.get('epoch')}) | device {device}"
          + (" | EMA weights" if ck.get("ema") is not None else ""))
    va, te = run_split(targs, run)

    cache = os.path.join(out, "_preds.npz")
    if os.path.exists(cache) and not args.no_cache:
        z = np.load(cache); pv, pt = z["val"], z["test"]
        pt6 = z["test6"] if "test6" in z else None
        print(f"loaded cached predictions from {cache}")
    else:
        tf = build_transforms(targs["image_size"], "none", targs.get("norm", "half"))
        pv, _ = predict(model, va, tf, device, args.batch_size, args.num_workers, not args.no_amp, "val")
        pt, pt6 = predict(model, te, tf, device, args.batch_size, args.num_workers, not args.no_amp, "test")
        np.savez(cache, val=pv, test=pt, **({"test6": pt6} if pt6 is not None else {}))
    yv, yt = va["label"].to_numpy().astype(int), te["label"].to_numpy().astype(int)

    beta = targs.get("threshold_beta", 1.0)
    target = args.target_recall or targs.get("target_recall", 0.95)
    thr_f, _ = M.best_f1_threshold(yv, pv, beta)
    thr_r = M.threshold_at_recall(yv, pv, target)
    ops = {"thr_0.5": 0.5, f"thr_bestF{beta:g}(val)": thr_f, f"thr_recall{target:g}(val)": thr_r}
    op_rows = [dict(operating_point=k, **M.compute_all(yt, pt, t)) for k, t in ops.items()]
    m05, mf = op_rows[0], op_rows[1]

    keys = tuple(k for _, k, _ in TABLE)
    ci05 = M.bootstrap_ci(yt, pt, 0.5, keys=keys, n_boot=args.bootstrap) if args.bootstrap else {}
    cif = M.bootstrap_ci(yt, pt, thr_f, keys=keys, n_boot=args.bootstrap) if args.bootstrap else {}
    g = gmacs(model, targs["image_size"], device)

    fmt_ci = lambda ci, k, f: f"[{ci[k][0]:{f}}, {ci[k][1]:{f}}]" if ci else ""
    rows = [{"Metric": label, "@0.5": f"{m05[k]:{f}}", "95% CI @0.5": fmt_ci(ci05, k, f),
             f"@val-F1 thr ({thr_f:.3f})": "same" if k in THRESHOLD_FREE else f"{mf[k]:{f}}",
             "95% CI @val-F1 thr": "" if k in THRESHOLD_FREE else fmt_ci(cif, k, f)} for label, k, f in TABLE]
    rows.append({"Metric": "Parameters (M)", "@0.5": f"{n_params / 1e6:.2f}"})
    if g:
        rows.append({"Metric": "GMACs (1 image)", "@0.5": f"{g:.2f}"})
    table = pd.DataFrame(rows).fillna("")
    table.to_csv(os.path.join(out, "metrics_table.csv"), index=False)
    md = [f"# Test metrics: {os.path.basename(run)}", "",
          f"Checkpoint `{ckpt_name}` (epoch {ck.get('epoch')}), split `{targs['split_mode']}` seed {targs['seed']}, "
          f"test N = {len(yt)} ({int(yt.sum())} cracked, prevalence {yt.mean():.3f}). Positive class = Cracked.",
          "Thresholds are tuned on VAL and applied to TEST. ECE uses 15 equal-width bins."
          + (f" 95% CIs: {args.bootstrap} stratified bootstrap resamples." if args.bootstrap else ""), "",
          table.to_markdown(index=False, disable_numparse=True), ""]
    with open(os.path.join(out, "metrics_table.md"), "w", encoding="utf8") as f:
        f.write("\n".join(md))

    pd.DataFrame(op_rows).to_csv(os.path.join(out, "operating_points.csv"), index=False)
    srows = []
    for s in ("Decks", "Pavements", "Walls"):
        m = (te["surface"] == s).to_numpy()
        if m.sum():
            mm = M.compute_all(yt[m], pt[m], thr_f)
            srows.append(dict(surface=s, N=int(m.sum()), cracked=int(yt[m].sum()),
                              **{k: mm[k] for k in ("auc", "ap", "acc", "precision", "recall", "f1", "mcc")}))
    surf_df = pd.DataFrame(srows)
    surf_df.to_csv(os.path.join(out, "per_surface.csv"), index=False)
    pred = te[["path", "surface", "label"]].copy()
    pred["p_cracked"] = pt
    pred["pred@valF1thr"] = (pt >= thr_f).astype(int)
    pred.to_csv(os.path.join(out, "test_predictions.csv"), index=False)
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(dict(run=os.path.basename(run), ckpt=ckpt_name, epoch=ck.get("epoch"), total_params=n_params,
                       params_M=n_params / 1e6, gmacs=g, thr_val_f1=thr_f, thr_val_recall=thr_r,
                       test_at_0_5=m05, test_at_val_f1=mf, test_at_val_recall=op_rows[2], ci_at_0_5=ci05,
                       ci_at_val_f1=cif, per_surface=srows), f, indent=2, default=float)

    print("\n" + "\n".join(md[2:]))
    print("per surface (val-F1 threshold):\n" + surf_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if pt6 is not None:         # --task six: the CrackNeXt SDNET2018 metrics
        m6 = M.six_class(add_six_class(te)["cls6"].to_numpy(), pt6)
        cm = pd.DataFrame(m6.pop("cm6"), index=[f"true {c}" for c in SIX_CLASSES], columns=SIX_CLASSES)
        cm.to_csv(os.path.join(out, "confusion6.csv"))
        with open(os.path.join(out, "six_class.json"), "w") as f:
            json.dump(m6, f, indent=2)
        print(f"\n6-CLASS (argmax): acc {m6['acc6']:.4f} | macro precision {m6['precision_m6']:.4f} | "
              f"macro recall {m6['recall_m6']:.4f} | macro F1 {m6['f1m6']:.4f}\n" + cm.to_string())
    print(f"\nall outputs -> {out}")
    hist = ck.get("history") or (json.load(open(os.path.join(run, "history.json")))
                                 if os.path.exists(os.path.join(run, "history.json")) else None)
    write_report(run, os.path.basename(run), yv, pv, yt, pt, n_params, g, hist, ckpt_name, ck.get("epoch"))


@torch.no_grad()
def evaluate_epochs(args, run, device):
    """val + test metrics of every epoch_NNNN.pt (no bootstrap) -> <run>/eval_epochs.csv + figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = sorted(glob.glob(os.path.join(run, "epoch_*.pt")))
    if not files:
        raise SystemExit(f"no epoch_NNNN.pt in {run} (train with --save-every 1)")
    targs = json.load(open(os.path.join(run, "config.json")))["args"]
    va, te = run_split(targs, run)
    yv, yt = va["label"].to_numpy().astype(int), te["label"].to_numpy().astype(int)
    tf = build_transforms(targs["image_size"], "none", targs.get("norm", "half"))
    rows = []
    for f in tqdm(files, desc="epochs"):
        model, _, ck = load_run(run, os.path.basename(f), device)
        pv = predict(model, va, tf, device, args.batch_size, args.num_workers, not args.no_amp, "val")
        pt = predict(model, te, tf, device, args.batch_size, args.num_workers, not args.no_amp, "test")
        thr, _ = M.best_f1_threshold(yv, pv)
        mv, mt = M.compute_all(yv, pv, thr), M.compute_all(yt, pt, thr)
        rows.append(dict(epoch=ck["epoch"], thr_val_f1=thr,
                         **{f"val_{k}": mv[k] for k in ("auc", "ap", "acc", "f1", "mcc")},
                         **{f"test_{k}": mt[k] for k in ("auc", "ap", "acc", "f1", "recall", "precision", "mcc")}))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(run, "eval_epochs.csv"), index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, k in zip(ax, ("auc", "f1")):
        a.plot(df.epoch, df[f"val_{k}"], label=f"val {k}"); a.plot(df.epoch, df[f"test_{k}"], "--", label=f"test {k}")
        a.set_xlabel("epoch"); a.grid(alpha=.3); a.legend()
    ax[0].set_title("ROC-AUC per saved epoch"); ax[1].set_title("F1 @ val-tuned threshold")
    fig.tight_layout(); fig.savefig(os.path.join(run, "fig_eval_epochs.png"), dpi=150); plt.close(fig)
    best = df.loc[df.val_auc.idxmax()]
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nbest epoch by VAL ROC-AUC: {int(best.epoch)} (test AUC {best.test_auc:.4f}) -> "
          f"{os.path.join(run, 'eval_epochs.csv')}")


def main():
    args = get_args()
    run = os.path.normpath(args.run)
    device = pick_device(args.device)
    if args.ckpt == "all":
        evaluate_epochs(args, run, device)
    else:
        evaluate_one(args, run, args.ckpt, device)


if __name__ == "__main__":
    main()
