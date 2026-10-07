"""
Train a Vision Transformer FROM SCRATCH on SDNET2018 (binary: Cracked vs Non-cracked).

    python train.py                                   # ViT-S/16, balanced split (config.yaml)
    python train.py --summary-only                    # model summary only, no data needed
    python train.py --variant vit_ti16 --batch-size 128
    python train.py --stem conv --attention str1      # ViT-S/16 conv stem + STR-1 skeleton-graph attention
    python train.py --variant hyb_r26                 # ResNet-style conv + STR-1 attention hybrid
    python train.py --resume auto                     # continue <run>/last.pt (same flags as the run)
    python train.py --resume runs/<run>/epoch_0030.pt # restart from any saved epoch

Flags override the YAML (every key in config.yaml is a flag: `drop_path` -> `--drop-path`).
Everything is written to runs/<run_name>/:
    best.pt last.pt epoch_NNNN.pt   config.json model.txt log.csv history.json train_log.txt curves.png
    results.json results.csv operating_points.csv per_surface.csv test_predictions.csv
    paper/ (metrics table + ROC/PR, reliability, confusion and training-curve figures)
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import shutil
import sys
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it, *a, **k):
        return it

import metrics as M
from augment import MinorityBank, mix_batch, smote_batch
from data import build_loaders, data_summary, load_split
from paper import write_report
from modelinfo import model_report
from models import arch_tag, config_from_args, model_from_cfg, normalized

MINIMISE = {"val_loss"}
# flags that change the optimisation; a resume with different values is allowed but warned about
RESUME_CHECK = ("split_mode", "seed", "augment", "batch_size", "grad_accum", "lr", "min_lr", "weight_decay",
                "warmup_epochs", "epochs", "scheduler", "label_smoothing", "class_weights", "grad_clip", "ema_decay",
                "mixup_alpha", "cutmix_alpha", "imbalance", "monitor")


class ConsoleLog:
    """Copies everything printed to stdout into <run>/train_log.txt (appending, so resumed runs keep the
    earlier part). Output printed before the run folder exists is buffered and written on attach().
    Progress bars go to stderr and are not logged."""

    def __init__(self):
        self.term, self.buf, self.file = sys.stdout, [], None
        sys.stdout = self

    def attach(self, path):
        self.file = open(path, "a", encoding="utf8")
        self.file.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} | python {' '.join(sys.argv)}\n"
                        + "".join(self.buf))
        self.buf = None
        self.file.flush()

    def write(self, s):
        self.term.write(s)
        if self.file:
            self.file.write(s); self.file.flush()
        elif self.buf is not None:
            self.buf.append(s)

    def flush(self):
        self.term.flush()


# ------------------------------------------------------------------ args
def str2bool(v):
    return v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "y")


def nullable(cast):
    """CLI value converter: 'null' / 'none' -> None (e.g. --early-stop-patience null)."""
    return lambda x: None if str(x).lower() in ("none", "null") else cast(x)


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml")
    pre, _ = p.parse_known_args()
    with open(pre.config, encoding="utf8") as f:
        cfg = {k.replace("-", "_"): v for k, v in (yaml.safe_load(f) or {}).items()}
    for k, v in cfg.items():                    # one flag per YAML key, typed from its default
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            p.add_argument(flag, type=str2bool, nargs="?", const=True, default=v)
        elif v is None:
            p.add_argument(flag, default=None)
        else:
            p.add_argument(flag, type=nullable(type(v)), default=v)
    args = p.parse_args()
    for k, v in vars(args).items():             # keys whose YAML default is null arrive as strings
        if cfg.get(k, 0) is None and isinstance(v, str):
            if v.lower() in ("none", "null", ""):
                setattr(args, k, None)
            else:
                for cast in (int, float):
                    try:
                        setattr(args, k, cast(v)); break
                    except ValueError:
                        pass
    return args


def run_name(a, c):
    tag = ((f"_cut{a.cutmix_alpha:g}" if a.cutmix_alpha > 0 else "") + (f"_mix{a.mixup_alpha:g}" if a.mixup_alpha > 0 else "")
           + (f"_{a.imbalance}" if a.imbalance != "off" else "") + (f"_ema" if a.ema_decay > 0 else ""))
    return f"{arch_tag(c)}_bs{a.batch_size * a.grad_accum}_lr{a.lr:g}_{a.split_mode}_{a.augment}{tag}_s{a.seed}"


def pick_device(name):
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else
                        "cpu" if name == "auto" else name)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def set_rng_state(s):
    random.setstate(s["python"]); np.random.set_state(s["numpy"]); torch.set_rng_state(s["torch"].cpu())
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in s["cuda"]])


def eta(history, total, epoch):
    """Remaining time from the mean of the last 3 epoch durations (upper bound; early stopping may end sooner)."""
    secs = [h["sec"] for h in history[-3:]]
    left = (total - epoch) * (sum(secs) / len(secs))
    return f"{int(left // 3600)}h{int(left % 3600 // 60):02d}m"


# ------------------------------------------------------------------ early stopping / EMA
class EarlyStopping:
    """Stops when `monitor` has not improved by > min_delta for `patience` epochs (None = never)."""

    def __init__(self, monitor, patience, min_delta):
        self.monitor, self.patience, self.min_delta = monitor, patience, min_delta
        self.sign = -1.0 if monitor in MINIMISE else 1.0
        self.best, self.best_epoch, self.bad = -math.inf, 0, 0

    def update(self, value, epoch):
        """Returns (improved, should_stop)."""
        v = self.sign * value
        improved = v > self.best + self.min_delta
        if improved:
            self.best, self.best_epoch, self.bad = v, epoch, 0
        else:
            self.bad += 1
        return improved, self.patience is not None and self.bad >= self.patience

    @property
    def best_value(self):
        return self.sign * self.best

    def state_dict(self):
        return dict(best=self.best, best_epoch=self.best_epoch, bad=self.bad)

    def load_state_dict(self, s):
        self.best, self.best_epoch, self.bad = s["best"], s["best_epoch"], s["bad"]


class ModelEma:
    """Exponential moving average of the weights; decay warms up as min(decay, (1+n)/(10+n))."""

    def __init__(self, model, decay):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay, self.updates = decay, 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for e, m in zip(self.module.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m.detach(), alpha=1 - d)
            else:
                e.copy_(m)


def sync_after_resume(out, es, ck_epoch, resume_path, history, log_path, cols):
    """Make the run folder consistent after resuming from ANY epoch checkpoint.
    * best.pt is reset to the best epoch within the resumed history (a later, abandoned branch of the
      same run may have left a different best.pt behind);
    * log.csv is rewritten from the resumed history, so epochs after the resume point are not duplicated."""
    src = resume_path if es.best_epoch == ck_epoch else os.path.join(out, f"epoch_{es.best_epoch:04d}.pt")
    dst = os.path.join(out, "best.pt")
    if os.path.exists(src):
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copyfile(src, dst)
    elif es.best_epoch:
        print(f"[warn] {src} not found, so best.pt cannot be re-synced; keep per-epoch checkpoints (--save-every 1) "
              f"to avoid this")
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for h in history:
            w.writerow([f"{v:.6g}" if isinstance(v := h.get(c), float) else ("" if v is None else v) for c in cols])


# ------------------------------------------------------------------ optimisation helpers
def cosine_lr(step, total, warmup, base, min_lr):
    if step < warmup:
        return base * (step + 1) / warmup
    t = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * t))


def param_groups(model, wd):
    """AdamW groups: no weight decay on norms, biases, cls token, pos-embed, gammas, relative position bias."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or n in ("cls_token", "pos_embed") or n.endswith((".gamma", "rel_bias"))
         else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def bce_loss(logits, y, smoothing, cw):
    """Class-weighted, label-smoothed BCE (soft targets from MixUp/CutMix are supported)."""
    t = y * (1 - smoothing) + 0.5 * smoothing
    w = cw[0] + (cw[1] - cw[0]) * y          # interpolates the class weight for soft (mixed) targets
    return (F.binary_cross_entropy_with_logits(logits.float(), t, reduction="none") * w).mean()


@torch.no_grad()
def predict(model, loader, device, amp_dtype, use_amp, desc=None):
    """-> y (N,), P(Cracked) (N,), plain BCE loss."""
    model.eval()
    ys, logits = [], []
    for x, y in tqdm(loader, desc=desc, leave=False, disable=desc is None, dynamic_ncols=True):
        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            logits.append(model(x.to(device, non_blocking=True)).float().cpu())
        ys.append(y)
    y, lg = torch.cat(ys), torch.cat(logits)
    return y.numpy().astype(int), torch.sigmoid(lg).numpy(), F.binary_cross_entropy_with_logits(lg, y).item()


def complexity(model, device, amp_dtype, use_amp, size, bs=64):
    info = dict(total_params=model.param_report()["total_params"])
    model.eval()
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            model(torch.randn(1, 3, size, size, device=device))
        info["gmacs"] = fc.get_total_flops() / 2e9          # FLOPs / 2 = MACs (forward, 1 image)
    except Exception as e:
        print(f"[complexity] flop count skipped: {e}")
    if device.type == "cuda":
        xb = torch.randn(bs, 3, size, size, device=device)
        with torch.no_grad(), torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            for _ in range(5):
                model(xb)
            torch.cuda.synchronize(); t = time.time()
            for _ in range(20):
                model(xb)
            torch.cuda.synchronize()
        info["throughput_img_s"] = bs * 20 / (time.time() - t)
    return info


def plot_curves(hist, path, title):
    h = pd.DataFrame(hist)
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
    ax[0].plot(h.epoch, h.train_loss, label="train (smoothed, weighted)"); ax[0].plot(h.epoch, h.val_loss, label="val BCE")
    ax[0].set_title("Loss")
    for a, key, t in [(ax[1], "auc", "Val ROC-AUC"), (ax[2], "f1", "Val F1 (Cracked @0.5)")]:
        a.plot(h.epoch, h[f"val_{key}"], label=key); a.set_title(t)
    ax[2].plot(h.epoch, h.val_mcc, label="mcc")
    ax[3].plot(h.epoch, h.lr, label="lr"); ax[3].set_title("Learning rate"); ax[3].set_yscale("log")
    for a in ax:
        a.set_xlabel("epoch"); a.grid(alpha=.3); a.legend(fontsize=8)
    fig.suptitle(title, fontsize=10); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    args = get_args()
    args.imbalance = args.imbalance or "off"
    if args.imbalance not in ("off", "oversample", "smote"):
        raise SystemExit("--imbalance must be off | oversample | smote")
    if args.epoch_ckpt not in ("full", "weights"):
        raise SystemExit("--epoch-ckpt must be full | weights")
    if args.imbalance != "off" and args.class_weights:
        print(f"[note] --imbalance {args.imbalance} already rebalances the classes, so loss class weights are turned off")
        args.class_weights = False
    seed_all(args.seed)
    device = pick_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    mcfg = config_from_args(args)
    model = model_from_cfg(mcfg.to_dict()).to(device)
    name = run_name(args, mcfg)
    print(f">>> RUN {name}  (from scratch, no pretrained weights)")
    report = model_report(model, mcfg, device)
    print(report)
    if args.summary_only:
        return

    out = os.path.join(args.output_dir, name)
    if args.resume == "auto":
        last = os.path.join(out, "last.pt")
        args.resume = last if os.path.exists(last) else None
        print(f"--resume auto: {'continuing ' + last if args.resume else 'no last.pt yet, starting fresh'}")

    use_amp = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "model.txt"), "w", encoding="utf8") as f:
        f.write(name + "\n\n" + report + "\n")
    if isinstance(sys.stdout, ConsoleLog):
        sys.stdout.attach(os.path.join(out, "train_log.txt"))
    cfg_path = os.path.join(out, "config.json")
    if args.resume and os.path.exists(cfg_path):          # warn if the resume flags differ from the original run
        old = json.load(open(cfg_path))["args"]
        diff = {k: (old.get(k), getattr(args, k)) for k in RESUME_CHECK if old.get(k) != getattr(args, k)}
        if diff:
            print("[warn] resuming with different flags than the run was started with: "
                  + ", ".join(f"{k} {a} -> {b}" for k, (a, b) in diff.items()))
    with open(cfg_path, "w") as f:
        json.dump({"args": vars(args), "model": mcfg.to_dict()}, f, indent=2)

    # ---- data ----------------------------------------------------------------
    df = load_split(args.data_root, args.split_mode, args.seed, args.split_dir, args.val_frac, args.test_frac)
    if args.debug_subset:       # quick pipeline test: N images per split (both classes kept)
        df = df.groupby(["split", "label"], group_keys=False).head(args.debug_subset // 2)
    print("\n" + data_summary(df, args.split_mode, args.data_root))
    train_loader, val_loader, test_loader, (tr_df, va_df, te_df) = build_loaders(
        df, args.image_size, args.batch_size, args.num_workers, args.augment, pin_memory=device.type == "cuda",
        seed=args.seed, norm=args.norm, oversample=args.imbalance == "oversample")

    n_pos = int(tr_df["label"].sum()); n_neg = len(tr_df) - n_pos
    cw = torch.tensor([len(tr_df) / (2 * n_neg), len(tr_df) / (2 * n_pos)] if args.class_weights else [1., 1.],
                      device=device)
    print(f"\nclass weights: Non-cracked={cw[0]:.3f}  Cracked={cw[1]:.3f}  (train prevalence {n_pos / len(tr_df):.3f})")
    print(f"loaders: train {len(train_loader)} batches x {args.batch_size} | val {len(val_loader)} x "
          f"{args.batch_size * 2} | test {len(test_loader)} x {args.batch_size * 2} | augment={args.augment}")
    bank = MinorityBank(args.smote_bank) if args.imbalance == "smote" else None
    print(f"batch augmentation: cutmix_alpha={args.cutmix_alpha} mixup_alpha={args.mixup_alpha} | imbalance={args.imbalance}")

    # ---- optim ---------------------------------------------------------------
    opt = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr, betas=(0.9, 0.999))
    accum = max(1, args.grad_accum)
    steps_per_epoch = math.ceil(len(train_loader) / accum)
    total_steps, warmup = args.epochs * steps_per_epoch, args.warmup_epochs * steps_per_epoch
    plateau = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=args.plateau_factor, patience=args.plateau_patience, min_lr=args.min_lr)
        if args.scheduler == "plateau" else None)
    es = EarlyStopping(args.monitor, args.early_stop_patience, args.early_stop_min_delta)
    ema = ModelEma(model, args.ema_decay) if args.ema_decay > 0 else None
    eval_net = ema.module if ema else model          # the weights that are validated, kept as best and tested

    start_epoch, step, history = 1, 0, []
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        if "optimizer" not in ck:
            raise SystemExit(f"--resume: {args.resume} is a weights-only checkpoint (epoch_ckpt=weights); "
                             f"resume from last.pt or a full epoch checkpoint")
        if normalized(ck["model_cfg"]) != mcfg.to_dict():
            raise SystemExit("--resume: checkpoint model config differs from the current one "
                             "(use the flags the run was started with)")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        if ema is not None:
            if ck.get("ema") is not None:
                ema.module.load_state_dict(ck["ema"]); ema.updates = ck.get("ema_updates", 0)
            else:
                ema.module.load_state_dict(ck["model"])
        if ck.get("scaler"): scaler.load_state_dict(ck["scaler"])
        if plateau and ck.get("plateau"): plateau.load_state_dict(ck["plateau"])
        if ck.get("rng"): set_rng_state(ck["rng"])
        es.load_state_dict(ck["early_stop"]); step, history = ck["step"], ck["history"]
        start_epoch = ck["epoch"] + 1
        print(f"resumed from {args.resume} -> continuing at epoch {start_epoch} "
              f"(best {args.monitor} {es.best_value:.4f} @ epoch {es.best_epoch}, no-improve count {es.bad})")

    cols = ["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "val_bal_acc", "val_auc", "val_ap",
            "val_f1", "val_f2", "val_precision", "val_recall", "val_mcc", "sec", "img_s"]
    log_path = os.path.join(out, "log.csv")
    if args.resume:
        sync_after_resume(out, es, start_epoch - 1, args.resume, history, log_path, cols)
    else:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(cols)

    print(f"\ntraining: epochs={args.epochs} lr={args.lr} wd={args.weight_decay} warmup={args.warmup_epochs} "
          f"scheduler={args.scheduler} | monitor={args.monitor} patience={args.early_stop_patience} "
          f"min_delta={args.early_stop_min_delta} | amp={amp_dtype if use_amp else 'off'} | ema={args.ema_decay or 'off'}"
          f" | save_every={args.save_every} ({args.epoch_ckpt})\n")

    # ---- train ---------------------------------------------------------------
    run_start, stopped_early = time.time(), False
    for epoch in range(start_epoch, args.epochs + 1):
        model.train(); t0 = time.time()
        n = correct = 0; loss_sum = 0.0; lr = opt.param_groups[0]["lr"]
        opt.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False, dynamic_ncols=True)
        for it, (x, y) in enumerate(pbar):
            if plateau is None and it % accum == 0:
                lr = cosine_lr(step, total_steps, warmup, args.lr, args.min_lr)
                for g in opt.param_groups:
                    g["lr"] = lr
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if bank is not None:
                x, y = smote_batch(x, y, bank, args.smote_target)
            x, y = mix_batch(x, y, args.mixup_alpha, args.cutmix_alpha, args.mixup_prob, args.mixup_switch_prob)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(x)
            loss = bce_loss(logits, y, args.label_smoothing, cw)
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at epoch {epoch} iter {it}; lower --lr or keep --grad-clip on. "
                                 f"Resume from the last good checkpoint with --resume runs/{name}/last.pt")
            scaler.scale(loss / accum).backward()
            if (it + 1) % accum == 0 or (it + 1) == len(train_loader):
                if args.grad_clip:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); step += 1
                if ema is not None:
                    ema.update(model)
            loss_sum += loss.item() * y.size(0)
            correct += ((logits > 0).float() == (y >= 0.5).float()).sum().item(); n += y.size(0)
            pbar.set_postfix(loss=f"{loss_sum / n:.4f}", acc=f"{correct / n:.4f}", lr=f"{lr:.1e}",
                             gpu=f"{torch.cuda.max_memory_allocated() / 2**30:.1f}G" if device.type == "cuda" else "cpu")
        pbar.close()
        train_sec = time.time() - t0

        # ---- validate ----
        yv, pv, vloss = predict(eval_net, val_loader, device, amp_dtype, use_amp, f"  val {epoch}")
        mv = M.compute_all(yv, pv)
        row = dict(epoch=epoch, lr=lr, train_loss=loss_sum / n, train_acc=correct / n, val_loss=vloss,
                   **{f"val_{k}": mv[k] for k in ("acc", "bal_acc", "auc", "ap", "f1", "f2", "precision", "recall", "mcc")},
                   sec=time.time() - t0, img_s=n / train_sec)
        history.append(row)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([f"{row[c]:.6g}" if isinstance(row[c], float) else row[c] for c in cols])

        improved, stop = es.update(row[args.monitor], epoch)
        if plateau is not None:
            plateau.step(vloss)
        weights = {"model": model.state_dict(), "ema": ema.module.state_dict() if ema else None,
                   "model_cfg": mcfg.to_dict(), "epoch": epoch, "val": {k: row[k] for k in cols if k.startswith("val_")}}
        ck = {**weights, "ema_updates": ema.updates if ema else 0, "args": vars(args), "step": step,
              "optimizer": opt.state_dict(), "scaler": scaler.state_dict(), "history": history,
              "plateau": plateau.state_dict() if plateau else None, "early_stop": es.state_dict(), "rng": rng_state()}
        torch.save(ck, os.path.join(out, "last.pt"))
        if improved:
            torch.save(ck, os.path.join(out, "best.pt"))
        if args.save_every and epoch % args.save_every == 0:
            torch.save(ck if args.epoch_ckpt == "full" else weights, os.path.join(out, f"epoch_{epoch:04d}.pt"))
        with open(os.path.join(out, "history.json"), "w") as f:
            json.dump(history, f, indent=1)

        print(f"epoch {epoch:3d}/{args.epochs} | lr {lr:.2e} | train loss {row['train_loss']:.4f} acc {row['train_acc']:.4f}"
              f" | val loss {vloss:.4f} acc {mv['acc']:.4f} auc {mv['auc']:.4f} ap {mv['ap']:.4f} f1 {mv['f1']:.4f}"
              f" rec {mv['recall']:.4f} prec {mv['precision']:.4f} mcc {mv['mcc']:.4f}"
              f" | {row['sec']:.0f}s ({row['img_s']:.0f} img/s) | ETA {eta(history, args.epochs, epoch)}"
              f" | best {args.monitor} {es.best_value:.4f}"
              + ("  * best" if improved else f"  (no improv {es.bad}/{args.early_stop_patience or '-'})"))
        if stop:
            stopped_early = True
            print(f"[early stopping] {args.monitor} did not improve for {es.bad} epochs "
                  f"(best {es.best_value:.4f} @ epoch {es.best_epoch})")
            break

    del train_loader                    # free its persistent worker processes before the test loader starts
    import gc; gc.collect()
    total_min = (time.time() - run_start) / 60
    if not history:
        raise SystemExit("no epochs were run (is --epochs larger than the resumed epoch?)")
    print(f"\ntraining done in {total_min:.1f} min | best {args.monitor}={es.best_value:.4f} @ epoch {es.best_epoch}")

    # ================================================================ final evaluation (best.pt)
    ck = torch.load(os.path.join(out, "best.pt"), map_location=device, weights_only=False)
    eval_net.load_state_dict(ck["ema"] if ema is not None and ck.get("ema") is not None else ck["model"])
    yv, pv, _ = predict(eval_net, val_loader, device, amp_dtype, use_amp, "val")
    yt, pt, tloss = predict(eval_net, test_loader, device, amp_dtype, use_amp, "test")

    # thresholds are chosen on VAL only, then applied to TEST
    thr_f, _ = M.best_f1_threshold(yv, pv, args.threshold_beta)
    thr_r = M.threshold_at_recall(yv, pv, args.target_recall)
    ops = {"thr_0.5": 0.5, f"thr_bestF{args.threshold_beta:g}(val)": thr_f, f"thr_recall{args.target_recall:g}(val)": thr_r}
    op_rows = [dict(operating_point=k, **M.compute_all(yt, pt, t)) for k, t in ops.items()]
    op_df = pd.DataFrame(op_rows)
    show = ["operating_point", "threshold", "acc", "bal_acc", "precision", "recall", "specificity", "f1", "f2",
            "mcc", "iou_crack", "fn", "fp"]
    print("\n" + "=" * 78 + f"\n  TEST — {name} — best.pt (epoch {ck['epoch']})\n" + "=" * 78)
    m0 = op_rows[0]
    print(f"  ROC-AUC {m0['auc']:.4f} | PR-AUC {m0['ap']:.4f} | Brier {m0['brier']:.4f} | ECE {m0['ece']:.4f} | NLL {m0['nll']:.4f}")
    print(op_df[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    ci = M.bootstrap_ci(yt, pt, thr_f, n_boot=args.bootstrap) if args.bootstrap else {}
    if ci:
        print(f"\n  95% bootstrap CI @val-F1 thr ({thr_f:.3f}): "
              + " | ".join(f"{k} [{lo:.4f}, {hi:.4f}]" for k, (lo, hi) in ci.items()))

    surf = te_df["surface"].to_numpy()
    srows = []
    for s in ("Decks", "Pavements", "Walls"):
        m = surf == s
        if m.sum():
            mm = M.compute_all(yt[m], pt[m], thr_f)
            srows.append(dict(surface=s, N=int(m.sum()), cracked=int(yt[m].sum()),
                              **{k: mm[k] for k in ("auc", "ap", "acc", "precision", "recall", "f1", "mcc")}))
    surf_df = pd.DataFrame(srows)
    print("\n  PER-SURFACE (val-tuned F1 threshold)")
    print(surf_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    cx = complexity(eval_net, device, amp_dtype, use_amp, args.image_size)
    print("\n  COMPLEXITY: " + " | ".join(f"{k}={v:,.2f}" if isinstance(v, float) else f"{k}={v:,}" for k, v in cx.items()))

    # ---- write everything ----
    plot_curves(history, f"{out}/curves.png", name)
    op_df.to_csv(f"{out}/operating_points.csv", index=False)
    surf_df.to_csv(f"{out}/per_surface.csv", index=False)
    pred = te_df[["path", "surface", "label"]].copy()
    pred["p_cracked"] = pt
    pred.to_csv(f"{out}/test_predictions.csv", index=False)
    write_report(out, name, yv, pv, yt, pt, cx["total_params"], cx.get("gmacs"), history, "best.pt", ck["epoch"])

    res = dict(run=name, variant=mcfg.variant, arch=mcfg.arch, attention=mcfg.attention,
               **{k: getattr(mcfg, a, None) for k, a in (("dim", "dim"), ("depth", "depth"), ("heads", "num_heads"),
                                                         ("patch", "patch_size"), ("stem", "stem"), ("pool", "pool"),
                                                         ("attn_stages", "attn_stages"))},
               split_mode=args.split_mode, augment=args.augment,
               batch_size=args.batch_size * accum, lr=args.lr, epochs_run=len(history), best_epoch=es.best_epoch,
               stopped_early=stopped_early, train_min=total_min, test_loss=tloss, **cx, thr_val_f1=thr_f,
               thr_val_recall=thr_r)
    res.update({f"test_{k}": v for k, v in m0.items() if k != "threshold"})
    for k in ("acc", "precision", "recall", "specificity", "f1", "f2", "mcc", "iou_crack", "fn", "fp"):
        res[f"test_{k}@valF1thr"] = op_rows[1][k]
    for k, (lo, hi) in ci.items():
        res[f"ci_{k}_lo"], res[f"ci_{k}_hi"] = lo, hi
    for r in srows:
        res[f"{r['surface']}_auc"], res[f"{r['surface']}_f1"] = r["auc"], r["f1"]
    with open(f"{out}/results.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    pd.DataFrame([res]).to_csv(f"{out}/results.csv", index=False)
    print(f"\nall outputs -> {out}")
    print(f"Grad-CAM / attention maps:  python gradcam.py --run {out}")


if __name__ == "__main__":
    log = ConsoleLog()
    try:
        main()
    except BaseException:           # crashes and Ctrl+C end up in the log too
        if log.file:
            log.file.write(traceback.format_exc())
        raise
