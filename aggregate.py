"""
Collect results.json of every finished run into one comparison table (variants, ablations, seeds).

    python aggregate.py                      # runs/*/results.json -> runs/summary.md + runs/summary.csv
    python aggregate.py --root runs --filter balanced
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd

COLS = ["run", "variant", "total_params", "gmacs", "throughput_img_s", "epochs_run", "best_epoch", "train_min",
        "test_auc", "test_ap", "test_acc@valF1thr", "test_f1@valF1thr", "test_recall@valF1thr",
        "test_precision@valF1thr", "test_mcc@valF1thr", "test_brier", "test_ece"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="runs")
    ap.add_argument("--filter", default="", help="only runs whose name contains this text")
    ap.add_argument("--out", default=None, help="default: <root>/summary.md")
    a = ap.parse_args()
    recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.root, "*", "results.json")))]
    recs = [r for r in recs if a.filter in r["run"]]
    if not recs:
        raise SystemExit(f"no finished runs (results.json) under {a.root}")
    df = pd.DataFrame(recs)
    df = df[[c for c in COLS if c in df]].sort_values("test_auc", ascending=False)
    df["total_params"] = (df["total_params"] / 1e6).round(2)
    df = df.rename(columns={"total_params": "params_M"})
    out = a.out or os.path.join(a.root, "summary.md")
    df.to_csv(os.path.splitext(out)[0] + ".csv", index=False)
    with open(out, "w", encoding="utf8") as f:
        f.write("Test metrics of best.pt; thresholded metrics at the val-tuned F1 threshold.\n\n"
                + df.to_markdown(index=False, floatfmt=".4f") + "\n")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}") + f"\n-> {out}")


if __name__ == "__main__":
    main()
