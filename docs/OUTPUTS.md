# Output files

## `runs/<run>/` (train.py)

| file | content |
|---|---|
| `best.pt`, `last.pt`, `epoch_NNNN.pt` | checkpoints (see NOTES.md → Checkpoints) |
| `config.json` | all flags + model config |
| `train_log.txt` | everything printed (appended on resume, crashes included) |
| `log.csv`, `history.json` | per epoch: lr, train loss/acc, val loss/acc/bal-acc/AUC/AP/F1/F2/precision/recall/MCC, seconds, img/s |
| `curves.png` | loss, val AUC, val F1/MCC, learning rate |
| `operating_points.csv` | test metrics at threshold 0.5, at the val-best-F1 threshold and at the val 95%-recall threshold |
| `per_surface.csv` | Decks / Pavements / Walls |
| `test_predictions.csv` | P(Cracked) per test image |
| `results.json` / `.csv` | one row with everything (read by `aggregate.py`) |
| `paper/` | `metrics.md/.csv/.json`, `fig_training_curves.png`, `fig_roc_pr.png`, `fig_reliability.png`, `fig_confusion.png` |

## `evaluate.py`

`<run>/eval/` (or `eval_<ckpt>/`): `metrics_table.md/.csv` (with 95% bootstrap CIs), `operating_points.csv`,
`per_surface.csv`, `test_predictions.csv`, `metrics.json`; plus `<run>/paper/` (or `paper_<ckpt>/`).
`--ckpt all`: `<run>/eval_epochs.csv` and `fig_eval_epochs.png`. Choose epochs by validation only.

## `gradcam.py` → `<run>/gradcam/`

`fig_gradcam_cracked`, `fig_gradcam_cracked_vs_intact`, `fig_gradcam_failures`, `fig_faithfulness`
(png + pdf), `faithfulness.csv`, `faithfulness_curves.csv`, `test_predictions.csv` (with TP/FN/FP/TN case),
`summary.json`, and a generated `README.md` with the faithfulness table.

## Metrics

Positive class = Cracked. ROC-AUC and PR-AUC don't depend on a threshold. Every thresholded number uses the
threshold that maximises F1 on **validation**, applied to test. The paper table also includes accuracy, macro
P/R/F1, crack recall/precision, specificity, MCC, Brier score and ECE (15 bins).
