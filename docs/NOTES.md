# Notes — ViT from scratch on SDNET2018

See also: [STR1.md](STR1.md) — STR-1 skeleton-graph sparse attention in the ViT and in a ResNet-style hybrid
(`hybrid.py`), including the finding that the raw-gradient structure tensor follows texture and why the
default prior is a Hessian ridge prior.

## What was borrowed from the LoopCrackViT repo (`../../code repo`)

| here | from | changes |
|---|---|---|
| `data.py` | `data.py` (SDNET part) | added `augment: strong`; cached split paths are re-rooted if the dataset folder moves |
| `augment.py` | `augment.py` | unchanged (MixUp, CutMix, SMOTE-style minority bank) |
| `metrics.py` | `metrics.py` (binary part) | dropped multi-class and early-exit tables |
| `paper.py` | `paper.py` | "block applications" replaced by GMACs |
| `train.py` | `train.py` | single exit; ViT param groups; EMA; RNG state in checkpoints; `--resume auto`; warning when resume flags differ; weights-only epoch checkpoints option |
| `evaluate.py` | `evaluate.py` | per-surface table; `--ckpt all` evaluates every saved epoch |
| `gradcam.py` | `gradcam.py` | conv-stem / per-pass maps replaced by Grad-CAM at chosen blocks + attention rollout; writes its own README |
| `runs/_splits/*.csv` | `runs/_splits/` | copied, so train/val/test are the **same images** as the LoopCrackViT runs |

Kept as in the reference: YAML config where every key is a CLI flag, console log to `train_log.txt`,
early stopping on a validation metric, `best.pt` / `last.pt` / `epoch_NNNN.pt` every epoch, resume from any
epoch (best.pt and log.csv are re-synced to the resume point), thresholds tuned on validation only,
bootstrap CIs, paper table and figures, Grad-CAM figures (TP / TN vs TP / failures) and deletion/insertion
faithfulness.

## The model (`vit.py`)

Standard pre-norm ViT: patchify → [CLS] + learned position embedding → 12 × (LN → MHSA → DropPath, LN → MLP 4× GELU
→ DropPath) → LN → [CLS] → Linear → 1 logit (BCE, Cracked = 1). Uses `F.scaled_dot_product_attention`
(fused kernel); `gradcam.py` switches a block to explicit softmax attention only while explaining.

**ViT-S/16 matches timm exactly**: verified that the state-dict keys map one-to-one onto
`timm.create_model("vit_small_patch16_224", num_classes=1)` (only `patch_embed.proj` → `stem.proj`), the parameter
count is identical (21,666,049) and the outputs agree to 0.0 with copied weights. No pretrained weights are
ever loaded — everything is random-initialised (trunc-normal 0.02, timm/DeiT style).

### Variants

| variant | dim | depth | heads | patch | tokens | params | GMACs (224 px) |
|---|---|---|---|---|---|---|---|
| `vit_ti16` | 192 | 12 | 3 | 16 | 197 | 5.52 M | 1.07 |
| **`vit_s16`** | 384 | 12 | 6 | 16 | 197 | **21.67 M** | **4.24** |
| `vit_s32` | 384 | 12 | 6 | 32 | 50 | 22.49 M | 1.12 |
| `vit_m16` | 512 | 12 | 8 | 16 | 197 | 38.33 M | 7.51 |
| `vit_b16` | 768 | 12 | 12 | 16 | 197 | 85.80 M | 16.85 |
| `vit_b32` | 768 | 12 | 12 | 32 | 50 | 87.46 M | 4.36 |

For comparison, LoopCrackViT is 3.91 M parameters.

Architecture switches (all flags, all off by default so the main model stays a plain ViT-S/16):

* `--stem conv` — ViT_C conv stem (Xiao et al., 2021, "Early convolutions help transformers see better"):
  stride-2 3×3 conv + BN + ReLU ×4, then 1×1 to `dim`. Same 14×14 token grid. Usually trains more stably
  from scratch on small datasets; adds ~0.7 M params on S/16.
* `--pool gap` — mean of the patch tokens instead of [CLS].
* `--layer-scale 1e-4` — CaiT LayerScale.
* `--dim / --depth / --num-heads / --patch-size` — free-form sizes (the run name then records them).

### Which variants fit this desktop (RTX 3060 12 GB, bf16) — measured with `bench.py`, 2026-10-07

Peak VRAM / training img/s on synthetic data (GPU compute only; real epochs are slower if JPEG
decoding with `--num-workers 4` can't keep up — a real ViT-S/16 epoch ran at ~230 img/s).

| variant | batch 128 | batch 64 | batch 32 | suggested |
|---|---|---|---|---|
| `vit_ti16` | 2.2 GB / 971 | 1.2 GB / 941 | 0.6 GB / 837 | `--batch-size 128` |
| `vit_s16` | 4.5 GB / 433 | 2.4 GB / 414 | 1.4 GB / 381 | `--batch-size 64` (default) |
| `vit_s32` | 1.5 GB / 1497 | 0.9 GB / 1334 | 0.6 GB / 1025 | `--batch-size 128` |
| `vit_m16` | 6.1 GB / 276 | 3.3 GB / 272 | 1.9 GB / 247 | `--batch-size 64` |
| `vit_b16` | 9.4 GB / 148 | 5.3 GB / 138 | 3.2 GB / 112 | `--batch-size 64 --drop-path 0.2` |
| `vit_b32` | 3.4 GB / 447 | 2.3 GB / 330 | 1.7 GB / 237 | `--batch-size 64` |

Everything fits in 12 GB, even batch 128. ViT-S/16 on the balanced split (11,877 train images) is
roughly 1 min per epoch, so 100 epochs is about 1.5–2 h.

## Training recipe (defaults in `config.yaml`) and why

ViTs have no convolutional inductive bias, so from scratch on 11,877 balanced training images they
over-fit much more readily than a CNN or the 3.9 M LoopCrackViT. The defaults follow DeiT, scaled down:

| setting | value | reason |
|---|---|---|
| optimiser | AdamW, β = (0.9, 0.999) | standard for ViT |
| lr | 1e-4 peak, cosine to 1e-6 | 5e-4 stalled at chance (see the finding below); conv stem also works at 3e-4 |
| warm-up | 5 epochs linear | ViTs from scratch diverge without warm-up |
| weight decay | 0.05, not on norms / biases / cls / pos-embed | DeiT |
| drop_path | 0.1 (S/16) | DeiT; 0.2 for B, 0.0 for Ti |
| label smoothing | 0.05 | same as the reference |
| grad clip | 1.0 | stability |
| epochs / patience | 100 / 20 on val ROC-AUC | ViTs from scratch improve slowly and noisily; a short patience stops too early |
| AMP | bf16 | the 3060 supports bf16, so no GradScaler is needed |
| augment | `basic` (reference recipe) | keeps the comparison with LoopCrackViT fair |

### Finding (2026-10-07): the plain patch-16 ViT-S stalls at chance on SDNET

First full run (`vit_s16`, patch stem, lr 5e-4, batch 64, balanced split): after 11 epochs the train loss was
still ~0.693 and train accuracy ~0.51, i.e. chance; val AUC ~0.60. Memorisation test (516 balanced training
images, no augmentation or regularisation, 400 AdamW steps, batch 32) — training accuracy reached:

| model | lr 1e-4 | lr 3e-4 |
|---|---|---|
| ours ViT-S/16, patch stem | 0.71 | 0.57 |
| timm `vit_small_patch16_224` | 0.59 | 0.51 |
| ours ViT-S/16, `--stem conv` | **0.96** | **0.97** |

* Not a bug: timm's independent ViT-S/16 behaves the same.
* The patch-16 stem optimises badly from scratch on this data, and a higher LR makes it worse, hence lr 1e-4.
* The conv stem (Xiao et al., 2021) fixes optimisation; the 12 transformer blocks remain exactly ViT-S/16
  (+0.7 M params in the stem). It is the recommended main model; the plain patch-stem run is the baseline.
* Run names now include the LR (`..._bs64_lr0.0001_...`), so runs at different LRs don't overwrite each other.

Things to try, in order of expected benefit (one change per run, compare on **validation**):

1. `--augment strong` — RandAugment + RandomErasing. Likely the biggest gain for a from-scratch ViT.
   Erasing can remove a thin crack while the label stays Cracked; check val recall.
2. `--ema-decay 0.999` — weight EMA; val, test and best.pt then use the EMA weights.
3. `--mixup-alpha 0.2` / `--cutmix-alpha 1.0` — same CutMix caveat as in the reference repo.
4. `--epochs 200 --early-stop-patience 40` — from-scratch ViTs often keep improving late.
5. LR sweep with the conv stem: `--lr 3e-4`, `--lr 5e-5`.

Splits: `balanced` is the default (paper protocol; test N = 2,546). `--split-mode random` uses the full
imbalanced set like the LoopCrackViT SMOTE run (add `--imbalance smote` or keep `class_weights`).
`group` has no source-photo leakage and is the hardest. Results on different split modes are not comparable.

## Checkpoints, early stopping, resume

* After every epoch: `last.pt` (full state), `best.pt` when the monitor improves, and `epoch_NNNN.pt`
  (`--save-every 1`). A full checkpoint holds model + optimizer + scaler + scheduler + early-stopping state +
  history + RNG states (+ EMA): **~260 MB for S/16, ~1 GB for B/16**, so 100 epochs of S/16 ≈ 26 GB.
  `--epoch-ckpt weights` stores only the weights in `epoch_NNNN.pt` (~87 MB for S/16): those can be evaluated
  but not resumed from. `--save-every 5` is the other way to save disk.
* Early stopping: `--monitor` (val_auc, val_ap, val_f1, val_f2, val_mcc, val_bal_acc, val_recall, val_loss),
  `--early-stop-patience` (null = never), `--early-stop-min-delta`. The test set is never used for stopping,
  checkpoint choice or thresholds.
* `resume: auto` (the default) continues `runs/<run>/last.pt` if it exists (else starts fresh), so the same command can be
  re-run after a crash or reboot; `--resume null` forces a fresh start. `--resume <path>` restarts from any full checkpoint; `best.pt` and `log.csv`
  are rewritten to that point. Resume refuses a different model config and **warns** if training flags (lr,
  batch size, epochs, …) differ. Changing `--epochs` changes the cosine schedule from that point on.
* RNG states are restored on resume. The data loader's shuffle order is not, so a resumed run is close to an
  uninterrupted one but not bit-identical.

## Grad-CAM on a ViT (`gradcam.py`)

* The head reads only [CLS], so the patch tokens at the output of the last block get no gradient. Grad-CAM
  therefore uses the **input** tokens of a block — by default the last block and the middle one
  (`--layers 11 6`) — weighted by the spatially averaged gradient of the Cracked logit.
* **Attention rollout** (Abnar & Zuidema, 2020): head-averaged attention + identity, multiplied over all
  12 blocks, [CLS] row. It does not depend on the class.
* **Faithfulness**: deletion / insertion curves (Petsiuk et al., 2018) on 200 correctly classified cracked test
  images, each map against a random map, as in the reference `gradcam/README.md`. SDNET has no pixel masks,
  so localisation IoU cannot be computed.
* A 16-px patch grid gives 14×14 maps, which are coarser than the reference's 28×28 conv-stem map.

## Verified so far (2026-10-07, on CPU, because the GPU was busy)

* ViT-S/16 matches timm (keys, parameter count, outputs).
* Full pipeline on a tiny model (2 blocks, 64 px, 64 images per split): train → per-epoch checkpoints →
  early-stop bookkeeping → test evaluation → paper outputs; resume from `epoch_0002.pt` with log/best
  re-sync; `--resume auto`; the refusal to resume from a weights-only checkpoint; `evaluate.py` on the full
  balanced test split (2,546 images, the same count as the reference); `gradcam.py` (figures, faithfulness,
  `--images`); `aggregate.py`; conv stem, GAP pooling and LayerScale forward/backward on full-size S/16.
* GPU (2026-10-07): `bench.py` for all variants (table above); 2-epoch ViT-S/16 run on 512 real images per
  split with bf16 (2.4 GB at batch 64).
* **Not yet done (as of 2026-10-07):** a full training run. First numbers are in the next section.

## First full runs (2026-10-08, balanced split, lr 1e-4, all still in progress / unfinished)

The approach is named **Crack Skeleton-Graph Transformer (CSGT)** in the README; in the code it is STR-1.

| run | best val AUC (epoch) | val acc range | train acc | status |
|---|---|---|---|---|
| `vit_s16_conv_str1-soft-k8-d1_bs64_lr0.0001` (ViT-S/16 + conv stem + CSGT) | 0.916 (≤77) | low–mid 0.80s | 0.83 @ 77 | stopped at epoch 77; re-run the same command to continue |
| `hyb_r26-a0_bs64_lr0.0001` (plain ResNet-26, no attention) | 0.897 (26) | 0.80–0.83 | 0.79 @ 29 | running |
| `vit_s16_conv_bs64_lr0.0003` (dense-attention baseline) | — | — | — | started only |

Val AUC at matching epochs (CSGT ViT vs plain ResNet-26): epoch 10 0.732 vs 0.720, epoch 20 0.823 vs 0.879,
epoch 26 0.855 vs 0.897. The CNN learns faster early (stronger inductive bias); the ViT keeps improving late.

**Caveats for the paper**

* The CSGT run used lr 1e-4 but the dense baseline uses lr 3e-4, so they are **not a fair comparison**. Run
  `python train.py --stem conv --lr 0.0003 --attention str1` (COMMANDS.md §1); keep the lr 1e-4 run as an LR check.
* The conv stem (Xiao et al. 2021) is not our contribution and is what makes ViT-S trainable here (finding above),
  so CSGT's gain must be shown against **conv stem + dense attention**, not against the plain patch-stem ViT.
  Ablation grid: {patch, conv} stem × {dense, CSGT}; patch + CSGT tests whether CSGT alone rescues the patch stem
  (the skeleton graph is built from the input image, not from stem features).
* Compare models on **AUC / AP**, not accuracy at the 0.5 threshold: recall/precision swing ±10 points between
  epochs at a fixed threshold while AUC moves smoothly.
* Val acc > train acc (~3 points) is expected, not a bug: train acc is measured with drop-path and augmentation
  on, as a running average over the epoch; val uses the clean end-of-epoch network. A large (>10 point) gap that
  persists late would mean over-regularisation. Check: score the train set in eval mode with `best.pt`.

**Practical**

* Two runs fit on the 3060 together (~5 GB each of 12 GB), but one run already keeps the GPU at ~94%, so in
  parallel each runs at ~half speed and total time is the same. Run sequentially unless away.
* Windows PowerShell 5.1 drops `""` arguments: use `--hybrid-attn-stages none` (or `0`) for a plain ResNet.

## Getting above ~92% accuracy (plan, 2026-10-08)

92% accuracy on the **balanced** split, from scratch, is unlikely for any architecture: SDNET is limited by
data, not model (hairline cracks vs joints / shadows / texture, label noise, only 11,877 training images after
undersampling). Published 90%+ numbers are usually pretrained (ImageNet) or on the imbalanced split, where ~85%
of images are Non-cracked and accuracy is inflated. Realistic levers, by expected gain:

1. **Use all the data (biggest lever).** The balanced split drops 39,124 Non-cracked images *before* splitting.
   Train on the full ~39k-image training pool with class weights or a balanced sampler, keep the **test** set
   balanced so it stays comparable. Implemented as `--train-full` (2026-10-08): train grows to 51,001 images (11.6% cracked), val / test
   are the identical balanced images. Use with `--imbalance oversample` or the default class weights.
2. **Self-supervised pretraining on SDNET itself** (MAE or DINO on all 56k images, no labels), then fine-tune.
   Still "from scratch" (no external data); usually the largest single gain for ViTs on small data.
3. **Stronger recipe (flags already exist):** `--ema-decay 0.999`, `--epochs 200 --early-stop-patience 40`,
   `--mixup-alpha 0.2`, stronger augmentation, LR 3e-4 vs 1e-4. Expect ~1–3 points.
4. **Architectures with conv inductive bias:** CNNs (ResNet, ConvNeXt-T) and hybrids usually beat plain ViTs from
   scratch on small data; plain ResNet-26 is already ahead of the ViT at matching epochs, so
   **hybrid ResNet-26 + CSGT** (`--variant hyb_r26 --attention str1`) may be the best model.
5. **Cheap final gains:** ensemble of 3–5 seeds, test-time augmentation (flips), accuracy-optimal threshold
   instead of 0.5. ~0.5–1.5 points each.

Order: 1 + 3 first (~a day of GPU, helps every model) → hybrid + CSGT → 2 if a large jump is still needed.
The paper's headline claim should be "CSGT beats dense attention under the same recipe" (AUC), which is stronger
than any single accuracy number.
