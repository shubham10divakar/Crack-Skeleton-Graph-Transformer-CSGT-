# Commands

Copy-paste list of every run, in the order to run them. Run from this folder (`new approach\code_repo`).
How the code works: [README.md](README.md). STR-1 design: [docs/STR1.md](docs/STR1.md).

Every run prints the full model report (also saved to `runs/<run>/model.txt`), writes `best.pt`, `last.pt` and
`epoch_NNNN.pt` every epoch, stops early on val ROC-AUC (patience 20), then evaluates on test.
**If a run is interrupted, re-run the same command: it continues from `last.pt` (`resume: auto` is the default). Add `--resume null` to start over.**

## 0. Check before training (no GPU time)

```powershell
python train.py --summary-only --stem conv --attention str1          # full model report, ViT + STR-1
python train.py --summary-only --variant hyb_r26 --attention str1    # full model report, hybrid + STR-1
python bench.py --variants vit_s16 hyb_r26 hyb_r50 --stem conv --attention str1   # VRAM + speed
```

## 1. ViT-S/16 (conv stem): dense attention vs STR-1

Same LR in both, so the only difference is the attention.

```powershell
python train.py --stem conv --lr 0.0003                              # reference: dense MHSA
python train.py --stem conv --lr 0.0003 --attention str1             # STR-1 skeleton-graph attention
```

## 2. ResNet-style hybrid: no attention vs dense attention vs STR-1

```powershell
python train.py --variant hyb_r26 --hybrid-attn-stages none          # plain ResNet-26 (conv only)
python train.py --variant hyb_r26 --attention mhsa                   # BoTNet-style dense attention
python train.py --variant hyb_r26 --attention str1                   # STR-1 attention
python train.py --variant hyb_r50 --attention str1                   # bigger: ResNet-50 layout + STR-1
```

## 3. Plain ViT-S/16 baseline (patch stem; trains slowly from scratch, see docs/NOTES.md)

```powershell
python train.py
```

## 4. STR-1 ablations (run on whichever model wins in 1 / 2; shown for the ViT)

```powershell
python train.py --stem conv --lr 0.0003 --attention str1 --str-k 4
python train.py --stem conv --lr 0.0003 --attention str1 --str-k 16
python train.py --stem conv --lr 0.0003 --attention str1 --str-mode hard
python train.py --stem conv --lr 0.0003 --attention str1 --str-dense-heads 0
python train.py --stem conv --lr 0.0003 --attention str1 --str-coh-gate false
python train.py --stem conv --lr 0.0003 --attention str1 --str-prior gradient
python train.py --variant hyb_r26 --attention str1 --hybrid-attn-stages "4"    # BoTNet placement only
```

## 5. Seeds for the paper (mean ± s.d.) — for the final models

```powershell
python train.py --stem conv --lr 0.0003 --attention str1 --seed 43
python train.py --stem conv --lr 0.0003 --attention str1 --seed 44
python paper.py runs/vit_s16_conv_str1-soft-k8-d1_bs64_lr0.0003_balanced_basic_s42 runs/vit_s16_conv_str1-soft-k8-d1_bs64_lr0.0003_balanced_basic_s43 runs/vit_s16_conv_str1-soft-k8-d1_bs64_lr0.0003_balanced_basic_s44 --out runs/paper_vit_str1.md
```

## 6. After training: explanations, evaluation, comparison

```powershell
python gradcam.py --run runs/<run>                   # Grad-CAM, rollout / STR-1 graph, faithfulness -> <run>/gradcam/
python evaluate.py --run runs/<run>                  # re-score best.pt (bootstrap CIs) -> <run>/eval/
python evaluate.py --run runs/<run> --ckpt all       # every saved epoch -> <run>/eval_epochs.csv
python aggregate.py                                  # all finished runs -> runs/summary.md
```

## 7. Resume / restart

```powershell
<same command as the run>                                            # continue from last.pt (resume: auto is the default)
<same command as the run> --resume null                              # start over from epoch 1 (overwrites the run)
<same command as the run> --resume runs/<run>/epoch_0030.pt          # restart from a chosen epoch
```

## Useful extras

| need | add |
|---|---|
| less disk (checkpoints) | `--save-every 5` or `--epoch-ckpt weights` |
| GPU not fully used (data loading slow) | `--num-workers 6` (back to 4 on `MemoryError`) |
| stronger augmentation | `--augment strong` |
| full imbalanced dataset | `--split-mode random` |
| quick pipeline test | `--debug-subset 64 --epochs 2 --bootstrap 0 --output-dir runs_debug` |
