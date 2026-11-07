# Leaderboard

All runs so far: SDNET2018, **balanced split, seed 42** (`runs/_splits/split_balanced_seed42.csv`), 224 px, batch 64,
basic augmentation unless noted. Val/test ≈ 2.5k images each. Single seed — no error bars yet.

"Best ep" = the epoch `best.pt` was saved from (selected by the run's monitor metric, `val_auc` unless noted);
val metrics are from that epoch. Last updated 2026-11-07 (runs last trained 2026-10-08).

## Binary (cracked vs non-cracked)

| # | Model | Init | Params | Status | Best ep | Val acc | Val AUC | Val F1 | Val MCC | **Test acc** | Test AUC | Test F1 | Test MCC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Hybrid R26, no STR-1 (`hyb_r26-a0`) — baseline | scratch | 15.1M | **done** 100/100 | 82 | 86.37 | 0.9311 | 0.8649 | 0.7274 | **87.20** | 0.9399 | 0.8738 | 0.7442 |
| 2 | **CSGT ViT-S/16** (STR-1 soft k8 d1, EMA, patch stem) | ImageNet-1k | 21.7M | stopped 18/50 | 16 | **87.07** | **0.9430** | 0.8695 | 0.7416 | — | — | — | — |
| 3 | **CSGT hybrid R26** (STR-1 in stages 3-4, EMA, lr 1e-3) | scratch | 13.1M | stopped 61/150 | 61 | 86.01 | 0.9353 | 0.8608 | 0.7203 | — | — | — | — |
| 4 | **CSGT ViT-S/16** (conv stem, lr 1e-4) | scratch | 22.4M | stopped 77/100 | 75 | 84.09 | 0.9157 | 0.8383 | 0.6821 | — | — | — | — |

Test (row 1, thr 0.5): precision 0.861, recall 0.887, specificity 0.857, AP 0.947, ECE 0.061; 2.80 GMACs.

## 6-class (CrackNeXt protocol: `--task six --augment cracknext --optimizer sgd`, lr 0.01, wd 2e-5, EMA)

Competitor: CrackNeXt **90.67%** 6-class acc (pretrained) — see protocol caveats in NOTES.md.

| # | Model | Init | Params | Status | Best ep (monitor `val_acc6`) | Val acc6 | Val macro-F1 (6) | Val binary acc | Val binary AUC | Test acc6 |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | **CSGT ViT-S/16** | ImageNet-1k | 21.7M | stopped 29/400 | 26 | 85.58 | 0.8535 | 86.40 | 0.9342 | — |

## Not run yet / no results

- Plain ViT-S/16 baseline without STR-1 (`vit_s16_conv`, lr 3e-4) — needed to isolate STR-1's gain on the ViT side.
- CSGT hybrid on train+val (`--train-full`) — interrupted in epoch 1.
- Seeds 43 / 44 for every row.

## Takeaways so far

- No CSGT run beats the plain hybrid baseline yet; pretrained CSGT ViT is level on val acc and best on val AUC at only 18/50 epochs.
- ImageNet init is worth ~+3 pts val acc for the ViT (row 2 vs row 4) in a fraction of the epochs.
- 6-class run is 5 pts short of CrackNeXt at 29/400 epochs — far too early to judge.

## Reproducing

Each run folder keeps `config.json` (exact args), `model.txt`, `log.csv`, `history.json`, `train_log.txt`,
`best.pt` (evaluate) and `last.pt` (resume: rerun the same `train.py` command — `--resume auto` continues it).
Per-epoch `epoch_*.pt` snapshots are not kept.
