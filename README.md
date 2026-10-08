# ViT from scratch on SDNET2018 — how to run

Plain ViT (main model: **ViT-S/16**, identical to timm `vit_small_patch16_224`), trained from scratch for
Cracked vs Non-cracked. Notes, recipe and design decisions: [docs/NOTES.md](docs/NOTES.md). STR-1 skeleton-graph attention
(ViT and ResNet-style hybrid): [docs/STR1.md](docs/STR1.md).
Output files: [docs/OUTPUTS.md](docs/OUTPUTS.md). **All experiment commands in order: [COMMANDS.md](COMMANDS.md).**

The dataset is read from the LoopCrackViT folder (`../../code repo/Structural Defects Network (SDNET) 2018 archive`),
set in `config.yaml` → `data_root`. Splits are the same CSVs as that repo (`runs/_splits/`).

**Our approach: Crack Skeleton-Graph Transformer (CSGT).** A crack is a graph (a skeleton with branches), so each
token attends only along plausible skeleton edges. This is STR-1 skeleton-graph sparse attention in the code
(`--attention str1`); main model: ViT-S/16, conv stem + STR-1 in all 12 blocks (`--stem conv --attention str1`).

```powershell
# 0. check the model / pipeline (no GPU time)
python train.py --summary-only
python train.py --debug-subset 64 --epochs 2 --bootstrap 0 --output-dir runs_debug

# 1. train ViT-S/16 (balanced split, defaults in config.yaml)
python train.py --stem conv                                      # recommended main model (see NOTES finding)
python train.py                                                  # plain patch-stem ViT-S/16 baseline

# 1b. STR-1 skeleton-graph sparse attention (docs/STR1.md)
python train.py --stem conv --attention str1                     # ViT-S/16 conv stem + STR-1
python train.py --variant hyb_r26 --attention str1               # ResNet-style conv + STR-1 attention hybrid
python train.py --variant hyb_r26 --attention mhsa               # same hybrid, dense attention (baseline)
python train.py --variant hyb_r26 --hybrid-attn-stages ""        # same hybrid, no attention (plain ResNet)

# 2. other variants
python train.py --variant vit_ti16 --batch-size 128
python train.py --variant vit_s32  --batch-size 128
python train.py --variant vit_b16  --batch-size 64 --drop-path 0.2

# 3. resume (same flags as the original run)
python train.py                                                  # re-run the same command: continues from last.pt
python train.py --resume null                                    # start over from epoch 1 (overwrites the run)
python train.py --resume runs/<run>/epoch_0030.pt                # restart from any epoch

# 4. evaluate a checkpoint / every saved epoch
python evaluate.py --run runs/<run>                              # best.pt -> <run>/eval + <run>/paper
python evaluate.py --run runs/<run> --ckpt last.pt
python evaluate.py --run runs/<run> --ckpt all                   # -> <run>/eval_epochs.csv

# 5. Grad-CAM + attention rollout + faithfulness
python gradcam.py --run runs/<run>                               # -> <run>/gradcam/
python gradcam.py --run runs/<run> --images my1.jpg my2.jpg

# 6. compare runs / seeds
python aggregate.py                                              # -> runs/summary.md
python paper.py runs/<run_s42> runs/<run_s43> runs/<run_s44> --out runs/paper_mean_std.md

# measure VRAM / speed of every variant on this GPU (run while the GPU is idle)
python bench.py
```

Every key in `config.yaml` is also a flag (`drop_path` → `--drop-path`). Run folders are named
`<model>[_str1-...]_bs<batch>_lr<lr>_<split>_<augment>[tags]_s<seed>`, e.g.
`runs/vit_s16_conv_str1-soft-k8-d1_bs64_lr0.0001_balanced_basic_s42/`.
Every run prints the full model report (config, layer tree with output shapes and parameters, totals, GMACs)
and saves it to `<run>/model.txt`. `python train.py --summary-only <flags>` prints it without training.

| problem | fix |
|---|---|
| out of GPU memory | lower `--batch-size`, raise `--grad-accum` (effective batch = product) |
| `MemoryError` / DataLoader worker died | `--num-workers 2` |
| `checkpoint model config differs` | resume with the model flags the run was started with |
| loss becomes NaN | lower `--lr`, keep `--grad-clip 1.0`, resume from the last good epoch |
