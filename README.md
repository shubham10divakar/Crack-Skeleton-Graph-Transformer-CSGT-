# ViT from scratch on SDNET2018 — how to run

Plain ViT (main model: **ViT-S/16**, identical to timm `vit_small_patch16_224`), trained from scratch for
Cracked vs Non-cracked. Notes, recipe and design decisions: [docs/NOTES.md](docs/NOTES.md).
Output files: [docs/OUTPUTS.md](docs/OUTPUTS.md).

The dataset is read from the LoopCrackViT folder (`../../code repo/Structural Defects Network (SDNET) 2018 archive`),
set in `config.yaml` → `data_root`. Splits are the same CSVs as that repo (`runs/_splits/`).

```powershell
# 0. check the model / pipeline (no GPU time)
python train.py --summary-only
python train.py --debug-subset 64 --epochs 2 --bootstrap 0 --output-dir runs_debug

# 1. train ViT-S/16 (balanced split, defaults in config.yaml)
python train.py --stem conv                                      # recommended main model (see NOTES finding)
python train.py                                                  # plain patch-stem ViT-S/16 baseline

# 2. other variants
python train.py --variant vit_ti16 --batch-size 128
python train.py --variant vit_s32  --batch-size 128
python train.py --variant vit_b16  --batch-size 64 --drop-path 0.2

# 3. resume (same flags as the original run)
python train.py --resume auto                                    # continue from last.pt
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
`<variant>[_conv][_gap]_bs<batch>_<split>_<augment>[tags]_s<seed>`, e.g. `runs/vit_s16_bs64_balanced_basic_s42/`.

| problem | fix |
|---|---|
| out of GPU memory | lower `--batch-size`, raise `--grad-accum` (effective batch = product) |
| `MemoryError` / DataLoader worker died | `--num-workers 2` |
| `checkpoint model config differs` | resume with the model flags the run was started with |
| loss becomes NaN | lower `--lr`, keep `--grad-clip 1.0`, resume from the last good epoch |
