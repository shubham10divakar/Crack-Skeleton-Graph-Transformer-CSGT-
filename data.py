"""
SDNET2018 data pipeline (binary: Cracked = 1 / Non-cracked = 0). Borrowed from the LoopCrackViT repo
(`code repo/data.py`), SDNET part only, plus a `strong` augmentation level for from-scratch ViTs.

Layout: <root>/{Decks,Pavements,Walls}/{Cracked,Non-cracked}/<image>-<patch>.jpg

SDNET files are patches cut from larger photos ("7001-115.jpg" = photo 7001, patch 115).
A random patch-level split therefore leaks near-identical neighbouring patches between
train and test. split_mode:
    balanced          - CrackNeXt protocol: per surface, undersample Non-cracked to the Cracked count
                        (~17k images, 50/50), then the patch-level random split (the paper protocol).
                        With --train-full the dropped Non-cracked images are added to train only
                        (val / test unchanged), see add_full_train()
    random            - patch-level split on the full imbalanced set (56,092 images, 15.1% cracked)
    group             - all patches of one source photo stay in the same split (no leakage; hardest)
All are stratified on surface x label, 70/15/15. The split is cached to a CSV so every run of the
same seed/mode sees exactly the same images. runs/_splits/ ships the CSVs of the LoopCrackViT repo,
so results are on identical images.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

SURFACES = ["Decks", "Pavements", "Walls"]
LABELS = {"Non-cracked": 0, "Cracked": 1}
CLASS_NAMES = ["Non-cracked", "Cracked"]
NORMS = {"half": ((0.5,) * 3, (0.5,) * 3),                       # from-scratch default
         "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))}
AUG_LEVELS = ("none", "basic", "strong")


def scan_sdnet(root):
    rows = []
    for s in SURFACES:
        for lab, y in LABELS.items():
            folder = os.path.join(root, s, lab)
            if not os.path.isdir(folder):
                raise FileNotFoundError(f"{folder}\n(set data_root in config.yaml or pass --data-root)")
            for f in os.listdir(folder):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    rows.append(dict(path=os.path.join(folder, f), label=y, surface=s,
                                     group=f"{s}_{f.rsplit('-', 1)[0]}"))
    df = pd.DataFrame(rows).sort_values("path").reset_index(drop=True)
    df["strat"] = df["surface"] + "_" + df["label"].astype(str)
    return df


def undersample(df, seed=42):
    """Within each surface, randomly keep as many Non-cracked images as there are Cracked ones
    (the CrackNeXt SDNET2018 protocol). Balancing happens BEFORE the train/val/test split."""
    keep = []
    for _, g in df.groupby("surface"):
        pos, neg = g[g["label"] == 1], g[g["label"] == 0]
        keep += [pos, neg.sample(n=min(len(pos), len(neg)), random_state=seed)]
    return pd.concat(keep).sort_values("path").reset_index(drop=True)


def make_split(df, mode="balanced", seed=42, val_frac=0.15, test_frac=0.15):
    if mode == "balanced":
        return make_split(undersample(df, seed), "random", seed, val_frac, test_frac)
    if mode == "random":
        tr, tmp = train_test_split(df, test_size=val_frac + test_frac, random_state=seed, stratify=df["strat"])
        va, te = train_test_split(tmp, test_size=test_frac / (val_frac + test_frac), random_state=seed,
                                  stratify=tmp["strat"])
        out = df.copy()
        out["split"] = "train"
        out.loc[va.index, "split"] = "val"
        out.loc[te.index, "split"] = "test"
        return out
    if mode != "group":
        raise ValueError("split_mode must be 'balanced', 'random' or 'group'")
    k = int(round(1 / min(val_frac, test_frac)))          # 15% -> ~7 folds
    folds = np.full(len(df), -1)
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    for i, (_, idx) in enumerate(sgkf.split(df, df["strat"], df["group"])):
        folds[idx] = i
    out = df.copy()
    out["split"] = "train"
    out.loc[folds == 0, "split"] = "test"
    out.loc[folds == 1, "split"] = "val"
    return out


def load_split(root, mode, seed, cache_dir, val_frac=0.15, test_frac=0.15):
    """Cached split. The cached CSV stores absolute paths; if the dataset moved, the paths are re-rooted onto
    `root` (same relative <surface>/<label>/<file>), so the same images are used without re-splitting."""
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"split_{mode}_seed{seed}.csv")
    if os.path.exists(cache):
        df = pd.read_csv(cache)
        if not os.path.exists(df["path"].iloc[0]):
            tail = lambda p: os.path.join(*os.path.normpath(p).replace("\\", "/").split("/")[-3:])
            df["path"] = [os.path.join(root, tail(p)) for p in df["path"]]
        if os.path.exists(df["path"].iloc[0]):
            return df
        print(f"[data] cached split {cache} does not match data_root; re-scanning")
    df = make_split(scan_sdnet(root), mode, seed, val_frac, test_frac)
    df.to_csv(cache, index=False)
    return df


def add_full_train(df, root):
    """--train-full: add every image the balanced split dropped (undersampled Non-cracked) to TRAIN only.
    val / test stay exactly the cached balanced images, so test numbers remain comparable with balanced runs.
    The train set becomes ~15% cracked; rebalance with class_weights (default) or --imbalance oversample.
    Images are matched on <surface>/<label>/<file>, so a moved data_root still lines up."""
    key = lambda d: d["surface"] + "/" + d["label"].astype(str) + "/" + d["path"].map(os.path.basename)
    full = scan_sdnet(root)
    extra = full[~key(full).isin(set(key(df)))].copy()
    extra["split"] = "train"
    return pd.concat([df, extra[df.columns]]).sort_values("path").reset_index(drop=True)


def build_transforms(size, augment="basic", norm="half"):
    """none   : resize only (val / test)
    basic  : the LoopCrackViT recipe - flips, affine (20 deg, 15% shift, 0.85-1.15 scale), brightness
    strong : basic + RandAugment(2, 9) + RandomErasing(p=0.25). ViTs trained from scratch on ~10-40k
             images over-fit fast; this is closer to the DeiT recipe. Erasing can hide a thin crack while
             the label stays Cracked, so treat it as an ablation (compare on validation)."""
    if augment is True:
        augment = "basic"
    if augment in (False, None):
        augment = "none"
    if augment not in AUG_LEVELS:
        raise ValueError(f"augment must be one of {AUG_LEVELS}")
    ev = [v2.Resize((size, size), antialias=True)]
    tail = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(*NORMS[norm])]
    if augment == "none":
        return v2.Compose(ev + tail)
    basic = [v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(),
             v2.RandomAffine(degrees=20, translate=(0.15, 0.15), scale=(0.85, 1.15)),
             v2.ColorJitter(brightness=(0.8, 1.2))]
    if augment == "basic":
        return v2.Compose(ev + basic + tail)
    return v2.Compose(ev + basic + [v2.RandAugment(num_ops=2, magnitude=9)] + tail
                      + [v2.RandomErasing(p=0.25, value=0)])


class CrackDataset(Dataset):
    def __init__(self, df, transform):
        self.paths = df["path"].tolist()
        self.labels = df["label"].to_numpy().astype("float32")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            img = self.transform(im.convert("RGB"))
        return img, self.labels[i]


def loader_kw(workers, persistent, pin_memory):
    """Windows spawns a full Python+torch process per worker (~1+ GB commit each), so keep the
    total small: train loader persistent, val loader 2 workers, test loader not persistent."""
    return dict(num_workers=workers, pin_memory=pin_memory, persistent_workers=persistent and workers > 0,
                prefetch_factor=4 if workers > 0 else None)


def build_loaders(df, image_size, batch_size, num_workers, augment="basic", pin_memory=True, seed=42,
                  norm="half", oversample=False):
    tr, va, te = (df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test"))
    gen = torch.Generator().manual_seed(seed)
    if oversample:      # every image drawn with probability ~ 1/(size of its class): batches are ~50% cracked
        counts = tr["label"].value_counts()
        wts = torch.tensor(tr["label"].map(lambda c: 1.0 / counts[c]).to_numpy(), dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(wts, num_samples=len(tr), replacement=True, generator=gen)
        order = dict(sampler=sampler)
    else:
        order = dict(shuffle=True, generator=gen)
    train_loader = DataLoader(CrackDataset(tr, build_transforms(image_size, augment, norm)),
                              batch_size=batch_size, drop_last=len(tr) > batch_size, **order,
                              **loader_kw(num_workers, True, pin_memory))
    ev_tf = build_transforms(image_size, "none", norm)
    val_loader = DataLoader(CrackDataset(va, ev_tf), batch_size=batch_size * 2, shuffle=False,
                            **loader_kw(min(2, num_workers), True, pin_memory))
    test_loader = DataLoader(CrackDataset(te, ev_tf), batch_size=batch_size * 2, shuffle=False,
                             **loader_kw(min(2, num_workers), False, pin_memory))
    return train_loader, val_loader, test_loader, (tr, va, te)


def data_summary(df, mode, root=None):
    """Printable dataset report: per split x surface counts with totals and crack prevalence.
    For split_mode 'balanced' it also shows the original folder counts and what undersampling dropped."""
    def table(d, by):
        t = d.groupby(by + ["label"]).size().unstack(fill_value=0).reindex(columns=[0, 1], fill_value=0)
        t.columns = ["Non-cracked", "Cracked"]
        t["Total"] = t.sum(1)
        t["Cracked %"] = (100 * t["Cracked"] / t["Total"]).round(1)
        return t

    order = ["train", "val", "test"]
    lines = [f"=== DATA  split_mode={mode} ===", "per split x surface:",
             table(df, ["split", "surface"]).reindex(order, level=0).to_string(), "", "per split:"]
    per = table(df, ["split"]).reindex(order)
    per["% of data"] = (100 * per["Total"] / len(df)).round(1)
    lines += [per.to_string(), "",
              f"TOTAL {len(df):,} images: {int((df.label == 0).sum()):,} Non-cracked + "
              f"{int(df.label.sum()):,} Cracked ({100 * df.label.mean():.1f}% cracked)"]
    if mode == "balanced+full-train":
        lines += ["", "train-full: the Non-cracked images undersampling dropped were added to TRAIN only; "
                      "val / test are the unchanged balanced splits"]
    if mode == "balanced" and root:
        try:
            full = scan_sdnet(root)
        except FileNotFoundError:
            return "\n".join(lines)
        orig, kept = table(full, ["surface"]), table(df, ["surface"])
        cmp = orig[["Non-cracked", "Cracked", "Total"]].add_prefix("orig ").join(
            kept[["Non-cracked", "Cracked", "Total"]].add_prefix("kept "))
        cmp["dropped Non-cracked"] = cmp["orig Non-cracked"] - cmp["kept Non-cracked"]
        lines += ["", "undersampling (per surface, Non-cracked cut to the Cracked count, before splitting):",
                  cmp.to_string(),
                  f"kept {len(df):,} of {len(full):,} images ({100 * len(df) / len(full):.1f}%); "
                  f"dropped {len(full) - len(df):,} Non-cracked"]
    return "\n".join(lines)
