"""
Grad-CAM and attention-rollout explanation figures for a trained SDNET2018 ViT, plus deletion / insertion
faithfulness. Adapted from the LoopCrackViT repo's gradcam.py.

    python gradcam.py --run runs/vit_s16_bs64_balanced_basic_s42             # -> <run>/gradcam/
    python gradcam.py --run runs/<run> --images a.jpg b.jpg                   # explain your own images

Panels per image:
    ViT     input | Grad-CAM last block | Grad-CAM middle block | attention rollout [| STR-1 coherence]
    hybrid  input | Grad-CAM stage 3 (14x14) | Grad-CAM stage 4 (7x7) [| STR-1 coherence]

For the hybrid (hybrid.py), Grad-CAM is the standard CNN one on the output feature map of a stage.
For STR-1 models the coherence prior c_i of the skeleton graph is shown too (it is a fixed input-derived prior,
not a learned saliency; its faithfulness row tells whether the geometry alone points at what the model uses),
and fig_str1_graph draws the token tangents and the graph edges of the most coherent tokens.

Grad-CAM on a ViT: with [CLS] pooling the head reads only [CLS], so the patch tokens at the OUTPUT of the
last block get no gradient. We therefore take the INPUT tokens of a block (its attention is what moves
patch evidence into [CLS]) - the last block (``--layers``, default: last and middle) - and back-propagate
the Cracked logit. Attention rollout (Abnar & Zuidema, 2020) multiplies the head-averaged attention
matrices (plus identity for the residual) over all blocks and reads the [CLS] row (or, for --pool gap,
the mean of the patch rows); it is class-agnostic.

Faithfulness (SDNET has no pixel masks, so localisation IoU is impossible): deletion / insertion curves
(Petsiuk et al., RISE 2018) on correctly classified cracked test images, every map vs a random map.
Lower deletion AUC and higher insertion AUC = the map points at what the model uses.

The decision threshold is tuned on VAL (best F1), exactly as in train.py.

Writes to <out> (default <run>/gradcam/):
    fig_gradcam_cracked.{png,pdf}            2 correctly detected cracks (TP) per surface
    fig_gradcam_cracked_vs_intact.{png,pdf}  intact (TN) vs cracked (TP), one per surface
    fig_gradcam_failures.{png,pdf}           a missed crack (FN) and a false alarm (FP) per surface
    fig_faithfulness.{png,pdf}               deletion / insertion curves
    fig_str1_graph.{png,pdf}                 STR-1 models: tangents (length ~ coherence) + skeleton-graph edges
    faithfulness.csv  faithfulness_curves.csv  test_predictions.csv  summary.json  README.md
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it, *a, **k):
        return it

import metrics as M
from data import NORMS, CrackDataset, build_transforms
from evaluate import load_run, pick_device, run_split
from hybrid import HybridNet

CASES = {"TP": "true positive", "FN": "false negative (missed crack)",
         "FP": "false positive", "TN": "true negative"}


def get_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--out", default=None, help="default: <run>/gradcam (or <run>/gradcam_<ckpt>)")
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="blocks whose INPUT tokens are used for Grad-CAM (default: last and depth//2)")
    ap.add_argument("--per-case", type=int, default=2, help="images per surface x case")
    ap.add_argument("--n-faith", type=int, default=200, help="images for deletion/insertion (0 = skip)")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--images", nargs="+", default=None, help="explain these image files only, then exit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    return ap.parse_args()


@torch.no_grad()
def predict(model, df, tf, device, bs):
    dl = DataLoader(CrackDataset(df, tf), batch_size=bs, shuffle=False, num_workers=0)
    return torch.cat([torch.sigmoid(model(x.to(device)).float()).cpu() for x, _ in tqdm(dl, desc="predict", leave=False)]).numpy()


# ---------------------------------------------------------------- explanations
class Explainer:
    """Grad-CAM at the input tokens of selected blocks + attention rollout, in one forward/backward."""

    def __init__(self, model, layers=None):
        self.m = model
        L = len(model.blocks)
        self.layers = sorted({(l % L) for l in (layers or [L - 1, L // 2])}, reverse=True)
        self.names = {l: ("last block" if l == L - 1 else f"block {l + 1}/{L}") for l in self.layers}
        self.tokens = {}
        for l in self.layers:
            model.blocks[l].register_forward_pre_hook(self._hook(l))

    def _hook(self, l):
        def fn(mod, args):
            x = args[0]
            if x.requires_grad:
                x.retain_grad()
            self.tokens[l] = x
        return fn

    @staticmethod
    def _cam(act, grad):
        """act/grad (B, C, h, w) -> (B, h, w) in [0, 1]. Channel weights = spatially averaged gradients."""
        cam = F.relu((grad.mean((2, 3), keepdim=True) * act).sum(1))
        return cam / cam.flatten(1).max(1)[0].clamp_min(1e-12)[:, None, None]

    def rollout(self):
        attn = [b.attn.attn.detach().float().mean(1) for b in self.m.blocks]       # (B, N, N) per block
        N = attn[0].shape[-1]
        eye = torch.eye(N, device=attn[0].device)
        R = eye.expand_as(attn[0]).clone()
        for a in attn:
            a = 0.5 * a + 0.5 * eye
            R = (a / a.sum(-1, keepdim=True)) @ R
        rows = R[:, 0, 1:] if self.m.cfg.pool == "cls" else R[:, 1:, 1:].mean(1)
        g = self.m.grid
        r = rows.reshape(-1, g, g)
        r = r - r.flatten(1).min(1)[0][:, None, None]
        return r / r.flatten(1).max(1)[0].clamp_min(1e-12)[:, None, None]

    def __call__(self, x):
        """x (B,3,H,W) normalised -> dict {map name: (B, g, g) numpy, 'p': (B,)}."""
        self.tokens = {}
        for b in self.m.blocks:
            b.attn.store_attn = True
        try:
            x = x.clone().requires_grad_(True)
            with torch.enable_grad():
                logit = self.m(x)
                logit.float().sum().backward()
            g = self.m.grid
            res = {}
            for l in self.layers:
                t = self.tokens[l]
                act = t[:, 1:].detach().float().transpose(1, 2).reshape(len(x), -1, g, g)
                grd = t.grad[:, 1:].float().transpose(1, 2).reshape(len(x), -1, g, g)
                res[f"gradcam_{l}"] = self._cam(act, grd)
            res["rollout"] = self.rollout()
            if self.m.graph is not None:
                res["coherence"] = self.m.graph(x.detach())["coh"].view(len(x), g, g)
        finally:
            for b in self.m.blocks:
                b.attn.store_attn, b.attn.attn = False, None
        out = {k: v.detach().cpu().numpy() for k, v in res.items()}
        out["p"] = torch.sigmoid(logit.detach().float()).cpu().numpy()
        return out

    def titles(self):
        t = {**{f"gradcam_{l}": f"Grad-CAM\n{self.names[l]}" for l in self.layers}, "rollout": "Attention\nrollout"}
        if self.m.graph is not None:
            t["coherence"] = "STR-1\ncoherence prior"
        return t

    def graph(self, x):
        return self.m.graph(x) if self.m.graph is not None else None


class HybridExplainer:
    """Grad-CAM on the output feature maps of stages 3 (14x14) and 4 (7x7) of HybridNet."""

    def __init__(self, model, stages=(3, 4)):
        self.m, self.stages, self.feats = model, list(stages), {}
        for s in self.stages:
            model.stages[s - 1][-1].register_forward_hook(self._hook(s))

    def _hook(self, s):
        def fn(mod, inp, out):
            if out.requires_grad:
                out.retain_grad()
                self.feats[s] = out
        return fn

    def __call__(self, x):
        self.feats = {}
        x = x.clone().requires_grad_(True)
        with torch.enable_grad():
            logit = self.m(x)
            logit.float().sum().backward()
        res = {}
        for s in self.stages:
            a, g = self.feats[s].detach().float(), self.feats[s].grad.float()
            res[f"gradcam_s{s}"] = Explainer._cam(a, g)
        geo = self.graph(x.detach())
        if geo is not None:
            res["coherence"] = geo["coh"].view(len(x), self.m.grid, self.m.grid)
        out = {k: v.detach().cpu().numpy() for k, v in res.items()}
        out["p"] = torch.sigmoid(logit.detach().float()).cpu().numpy()
        return out

    def _graph_module(self):
        g = self.m.graphs
        return g[str(self.m.grid)] if g is not None and str(self.m.grid) in g else None

    def graph(self, x):
        G = self._graph_module()
        return G(x) if G is not None else None

    def titles(self):
        sz = {3: self.m.cfg.image_size // 16, 4: self.m.cfg.image_size // 32}
        t = {f"gradcam_s{s}": f"Grad-CAM\nstage {s} ({sz[s]}x{sz[s]})" for s in self.stages}
        if self._graph_module() is not None:
            t["coherence"] = "STR-1\ncoherence prior"
        return t


def make_explainer(model, layers=None):
    return HybridExplainer(model) if isinstance(model, HybridNet) else Explainer(model, layers)


def draw_graph(rows, path, top=4):
    """STR-1 figure: tangent sticks (length ~ coherence) on every token + edges of the `top` most coherent tokens."""
    if not rows:
        return
    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4), squeeze=False)
    for ax, r in zip(axes[0], rows):
        img, th, c, E = r["img"], r["theta"], r["coh"], r["edges"]
        S, g = img.shape[0], int(round(len(th) ** 0.5))
        cell = S / g
        show_img(ax, img)
        ys, xs = np.divmod(np.arange(g * g), g)
        cx, cy = (xs + 0.5) * cell, (ys + 0.5) * cell
        L = 0.45 * cell * c
        ax.plot(np.stack([cx - L * np.cos(th), cx + L * np.cos(th)]),
                np.stack([cy - L * np.sin(th), cy + L * np.sin(th)]), color="#ffd400", lw=1.2)
        for i in np.argsort(-c)[:top]:
            for j in np.nonzero(E[i])[0]:
                if j != i:
                    ax.plot([cx[i], cx[j]], [cy[i], cy[j]], color="#00e5ff", lw=1.0, alpha=0.9)
            ax.scatter([cx[i]], [cy[i]], s=18, color="#ff2d55", zorder=3)
        ax.set_xlim(0, S); ax.set_ylim(S, 0)
        ax.set_xlabel(f"{r['label']}  p={r['p']:.2f}", fontsize=8)
    fig.suptitle("STR-1 skeleton graph: token tangents (yellow, length ~ coherence) and graph edges of the most "
                 "coherent tokens (cyan)", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def upsample(m, size):
    t = torch.from_numpy(np.ascontiguousarray(m))[None, None].float()
    return F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()


# ---------------------------------------------------------------- drawing
def show_img(ax, img):
    ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])


def overlay(ax, img, cam):
    show_img(ax, img)
    ax.imshow(upsample(cam, img.shape[0]), cmap="jet", alpha=0.45, vmin=0, vmax=1)


def draw_rows(rows, maps, titles, path, title=None):
    """rows: list of dicts with img, <map keys>, p, label text."""
    if not rows:
        print(f"[skip] {os.path.basename(path)}: no images for this figure")
        return
    n, c = len(rows), 1 + len(maps)
    fig, axes = plt.subplots(n, c, figsize=(c * 2.1, n * 2.25 + (0.5 if title else 0.2)), squeeze=False)
    for r, row in enumerate(rows):
        a = axes[r]
        show_img(a[0], row["img"])
        a[0].set_ylabel(row["label"], fontsize=8)
        for j, k in enumerate(maps, 1):
            overlay(a[j], row["img"], row[k])
        a[-1].set_xlabel(f"p(crack) = {row['p']:.2f}", fontsize=8)
        if r == 0:
            for j, t in enumerate(["Input"] + [titles[k] for k in maps]):
                a[j].set_title(t, fontsize=9)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def denorm(x, norm):
    m, s = (torch.tensor(v)[:, None, None] for v in NORMS[norm])
    return (x.cpu() * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


# ---------------------------------------------------------------- faithfulness
@torch.no_grad()
def deletion_insertion(model, x, maps, steps, bs):
    """x (N,3,H,W) normalised; maps (N,H,W) saliency at input size. Returns (del_curve, ins_curve) (steps+1,)."""
    N, _, H, W = x.shape
    base = F.avg_pool2d(F.pad(x, (5, 5, 5, 5), mode="reflect"), 11, 1)           # blurred image
    order = torch.from_numpy(maps).reshape(N, -1).argsort(1, descending=True)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(H * W).expand(N, -1))
    rank = rank.reshape(N, 1, H, W).to(x.device)
    dele, ins = [], []
    for k in range(steps + 1):
        top = rank < int(round(k / steps * H * W))
        for out, a, b in ((dele, base, x), (ins, x, base)):           # top pixels taken from a, rest from b
            img = torch.where(top, a, b)
            p = torch.cat([torch.sigmoid(model(img[i:i + bs]).float()) for i in range(0, N, bs)])
            out.append(p.mean().item())
    return np.array(dele), np.array(ins)


def trapz(y, dx):
    return float(np.sum((y[1:] + y[:-1]) * dx / 2))


# ---------------------------------------------------------------- main
def explain_files(model, expl, files, tf, norm, maps, titles, out):
    rows = []
    for f in files:
        with Image.open(f) as im:
            x = tf(im.convert("RGB"))
        e = expl(x[None].to(next(model.parameters()).device))
        rows.append({**{k: e[k][0] for k in maps}, "p": float(e["p"][0]), "img": denorm(x, norm),
                     "label": os.path.basename(f)})
        print(f"{f}: p(crack) = {e['p'][0]:.4f}")
    draw_rows(rows, maps, titles, os.path.join(out, "fig_gradcam_images"))
    print(f"-> {os.path.join(out, 'fig_gradcam_images.png')}")


def main():
    args = get_args()
    device = pick_device(args.device)
    rng = np.random.default_rng(args.seed)
    run = os.path.normpath(args.run)
    model, targs, ck = load_run(run, args.ckpt, device)
    for p in model.parameters():
        p.requires_grad_(False)                      # gradients are only needed w.r.t. the tokens
    norm = targs.get("norm", "half")
    tf = build_transforms(targs["image_size"], "none", norm)
    out = args.out or os.path.join(run, "gradcam" if args.ckpt == "best.pt" else f"gradcam_{os.path.splitext(args.ckpt)[0]}")
    os.makedirs(out, exist_ok=True)
    expl = make_explainer(model, args.layers)
    titles = expl.titles()
    maps = list(titles)

    if args.images:
        explain_files(model, expl, args.images, tf, norm, maps, titles, out)
        return

    va, te = run_split(targs, run)
    cache = os.path.join(out, "_preds.npz")
    if os.path.exists(cache):
        c = np.load(cache); pv, te["p_cracked"] = c["pv"], c["pt"]
    else:
        pv = predict(model, va, tf, device, args.batch_size)
        te["p_cracked"] = predict(model, te, tf, device, args.batch_size)
        np.savez(cache, pv=pv, pt=te["p_cracked"].to_numpy())
    thr, _ = M.best_f1_threshold(va["label"].to_numpy(), pv, 1.0)
    y, pr = te["label"].to_numpy(), (te["p_cracked"] >= thr).astype(int).to_numpy()
    te["pred"] = pr
    te["case"] = np.select([(y == 1) & (pr == 1), (y == 1) & (pr == 0), (y == 0) & (pr == 1)], ["TP", "FN", "FP"], "TN")
    te.to_csv(os.path.join(out, "test_predictions.csv"), index=False)
    summ = {"checkpoint": os.path.join(run, args.ckpt), "epoch": ck.get("epoch"), "threshold_val_bestF1": thr,
            "test_auc": roc_auc_score(y, te["p_cracked"]), "test_counts": te["case"].value_counts().to_dict(),
            "maps": [t.replace("\n", " ") for t in titles.values()]}
    print(json.dumps(summ, indent=2))

    # ---- examples: TP/TN = confident, FN/FP = most confidently wrong; sampled from the top 30
    picks = []
    for s in sorted(te["surface"].unique()):
        for c in CASES:
            d = te[(te["surface"] == s) & (te["case"] == c)]
            if d.empty:
                continue
            d = d.sort_values("p_cracked", ascending=c in ("FN", "TN")).head(30)
            picks.append(d.iloc[np.sort(rng.choice(len(d), min(args.per_case, len(d)), replace=False))])
    picks = pd.concat(picks)
    ds = CrackDataset(picks, tf)
    rows = []
    for i in tqdm(range(len(picks)), desc="grad-cam"):
        x = ds[i][0]
        e = expl(x[None].to(device))
        r = picks.iloc[i]
        rows.append({**{k: e[k][0] for k in maps}, "p": float(e["p"][0]), "img": denorm(x, norm),
                     "surface": r["surface"], "case": r["case"],
                     "label": f"{r['surface']} | {r['case']}\n{'Cracked' if r['label'] else 'Non-cracked'}"})

    by = lambda **kw: [r for r in rows if all(r[k] == v for k, v in kw.items())]
    surfaces = sorted({r["surface"] for r in rows})
    draw_rows([r for s in surfaces for r in by(surface=s, case="TP")[:2]], maps, titles,
              os.path.join(out, "fig_gradcam_cracked"),
              "Correctly detected cracks (TP): where the ViT finds crack evidence")
    draw_rows([r for s in surfaces for r in by(surface=s, case="TN")[:1]] +
              [r for s in surfaces for r in by(surface=s, case="TP")[:1]], maps, titles,
              os.path.join(out, "fig_gradcam_cracked_vs_intact"), "Intact (TN, top) vs cracked (TP, bottom)")
    draw_rows([r for s in surfaces for c in ("FN", "FP") for r in by(surface=s, case=c)[:1]], maps, titles,
              os.path.join(out, "fig_gradcam_failures"), "Failure cases: missed cracks (FN) and false alarms (FP)")
    if "coherence" in maps:                                    # STR-1 graph figure: one TP per surface
        mean, std = (np.array(v, dtype=np.float32) for v in NORMS[norm])
        grows = []
        for r in [r for s in surfaces for r in by(surface=s, case="TP")[:1]]:
            x = torch.from_numpy(((r["img"] - mean) / std).transpose(2, 0, 1).astype(np.float32))
            geo = expl.graph(x[None].to(device))
            grows.append(dict(img=r["img"], p=r["p"], label=r["surface"], theta=geo["theta"][0].cpu().numpy(),
                              coh=geo["coh"][0].cpu().numpy(), edges=geo["edges"][0].cpu().numpy()))
        draw_graph(grows, os.path.join(out, "fig_str1_graph"))

    fdf = None
    if args.n_faith > 0:
        tp = te[te["case"] == "TP"]
        tp = tp.iloc[np.sort(rng.choice(len(tp), min(args.n_faith, len(tp)), replace=False))]
        ds = CrackDataset(tp, tf)
        S = targs["image_size"]
        xs, cams = [], {k: [] for k in maps + ["random"]}
        for i in tqdm(range(0, len(tp), 16), desc="faithfulness maps"):
            x = torch.stack([ds[j][0] for j in range(i, min(i + 16, len(tp)))]).to(device)
            e = expl(x)
            for k in maps:
                cams[k] += [upsample(m, S) for m in e[k]]
            cams["random"] += [upsample(rng.random((model.grid, model.grid)), S) for _ in range(len(x))]
            xs.append(x.detach())
        x = torch.cat(xs)
        names = {**{k: titles[k].replace("\n", " ") for k in maps}, "random": "random map"}
        res, curves = [], {}
        for k in tqdm(names, desc="deletion/insertion"):
            d, i = deletion_insertion(model, x, np.stack(cams[k]), args.steps, args.batch_size)
            curves[k] = (d, i)
            res.append({"map": names[k], "deletion_auc(lower=better)": trapz(d, 1 / args.steps),
                        "insertion_auc(higher=better)": trapz(i, 1 / args.steps), "n_images": len(tp)})
        fdf = pd.DataFrame(res)
        fdf.to_csv(os.path.join(out, "faithfulness.csv"), index=False)
        pd.DataFrame({"fraction": np.linspace(0, 1, args.steps + 1),
                      **{f"{k}_{n}": c[j] for k, c in curves.items() for j, n in enumerate(("deletion", "insertion"))}}
                     ).to_csv(os.path.join(out, "faithfulness_curves.csv"), index=False)
        print(fdf.to_string(index=False))
        fr = np.linspace(0, 1, args.steps + 1)
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        for k, (d, i) in curves.items():
            ls = "--" if k == "random" else "-"
            ax[0].plot(fr, d, ls, label=names[k]); ax[1].plot(fr, i, ls, label=names[k])
        for a, t in zip(ax, ("Deletion (lower AUC = more faithful)", "Insertion (higher AUC = more faithful)")):
            a.set_title(t, fontsize=10); a.set_xlabel("fraction of pixels removed / inserted"); a.set_ylabel("mean p(crack)")
            a.grid(alpha=0.3)
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out, f"fig_faithfulness.{ext}"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        summ["faithfulness"] = res

    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)
    write_readme(out, run, args, summ, fdf, titles, model)
    print(f"\nall outputs -> {out}")


def write_readme(out, run, args, summ, fdf, titles, model):
    c = summ["test_counts"]
    md = [f"# Grad-CAM explanations — {os.path.basename(run)}", "",
          f"Generated by `python gradcam.py --run {run}` from `{args.ckpt}` (epoch {summ['epoch']}). "
          f"Test AUC {summ['test_auc']:.4f}; decision threshold {summ['threshold_val_bestF1']:.3f} tuned on val "
          f"(best F1); test cases TP {c.get('TP', 0)} / FN {c.get('FN', 0)} / FP {c.get('FP', 0)} / TN {c.get('TN', 0)}.",
          "", "| file | what |", "|---|---|",
          "| `fig_gradcam_cracked` | 2 correctly detected cracks per surface (Decks, Pavements, Walls) |",
          "| `fig_gradcam_cracked_vs_intact` | intact (TN) vs cracked (TP), one per surface |",
          "| `fig_gradcam_failures` | a missed crack (FN) and a false alarm (FP) per surface |",
          "| `fig_faithfulness` | deletion / insertion curves (`faithfulness.csv`, `faithfulness_curves.csv`) |",
          "| `fig_str1_graph` | STR-1 models: token tangents and skeleton-graph edges on one TP per surface |",
          "| `test_predictions.csv` | P(Cracked), prediction and TP/FN/FP/TN case per test image |", "",
          "Columns: input · " + " · ".join(t.replace("\n", " ") for t in titles.values()) + ". "
          + ("Grad-CAM on the output feature map of the named stage. " if isinstance(model, HybridNet) else
             "Grad-CAM uses the input tokens of the named block (the head reads only [CLS]); "
             "rollout multiplies head-averaged attention over all blocks. ")
          + ("The STR-1 coherence prior is computed from the input image (not learned); `fig_str1_graph` shows "
             "the token tangents and the skeleton-graph edges." if "coherence" in titles else ""), ""]
    if fdf is not None:
        md += [f"## Faithfulness ({int(fdf.n_images.iloc[0])} correctly classified cracked test images)", "",
               "| saliency map | deletion AUC ↓ | insertion AUC ↑ |", "|---|---|---|"]
        md += [f"| {r['map']} | {r['deletion_auc(lower=better)']:.3f} | {r['insertion_auc(higher=better)']:.3f} |"
               for _, r in fdf.iterrows()]
        md += ["", "A map is faithful if it beats the random map on both measures."]
    with open(os.path.join(out, "README.md"), "w", encoding="utf8") as f:
        f.write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
