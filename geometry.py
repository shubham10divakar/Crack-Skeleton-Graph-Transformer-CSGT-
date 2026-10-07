"""
STR-1 skeleton-graph sparse attention: geometric primitives and graph construction (design notes:
docs/STR1.md). Shared by the ViT (vit.py) and the ResNet-style hybrid (hybrid.py).

Per image, computed on the GPU from the network INPUT (i.e. after augmentation, so the prior always agrees
with flips / rotations / CutMix), without gradients. Two priors (`str_prior`):

  ridge (default)  Hessian of the grey image smoothed at crack scale sigma (2 px): per pixel, l_max >= l_min and
                   the normal psi (direction of l_max). Dark-line strength r = relu(l_max - |l_min|): positive for
                   a dark valley (crack), ~0 for dark blobs/pits (both curvatures large) and bright lines.
                   Token tensor T = avgpool_token( r * [cos^2 psi, cos psi sin psi, sin^2 psi] ), so only line
                   pixels vote on the token orientation; token ridge energy e_i = avgpool_token(r).
  gradient         plain structure tensor T = avgpool_token([gx^2, gx gy, gy^2]) (Sobel). On SDNET this is
                   dominated by aggregate texture inside a 16x16 patch (see docs/STR1.md); kept as an ablation.

    tangent        theta_i = 0.5 * atan2(2 Txy, Txx - Tyy) + pi/2        (direction ALONG the line, mod pi)
    coherence      coh_i = ((l1 - l2) / (l1 + l2 + eps))^2  in [0, 1]    (1 = one dominant orientation)
    gate           c_i = coh_i * e_i / max_j e_j      (ridge prior; = coh_i for the gradient prior)

Skeleton graph on the token grid (positions p_i, displacement D_ij = p_j - p_i, distance d_ij):

    cone        a_ij = |cos(angle(D_ij) - theta_i)|        displacement runs along i's tangent
    agreement   o_ij = |cos(theta_i - theta_j)|            j has a similar orientation
    edge prior  g_ij = 1[a_ij >= cos alpha] * 1[o_ij >= cos beta]                    (mode "hard")
                g_ij = sigmoid((a_ij - cos alpha)/tau) * sigmoid((o_ij - cos beta)/tau) (mode "soft")
    score       s_ij = c_i * log(g_ij + eps) - d_ij / rho        (c_i := 1 if gating is off)
    edges       E_i  = {i} U top-k_j { s_ij : 0 < d_ij <= radius }

Coherence gating makes the graph degrade gracefully: for a coherent (line-like) query the log-prior dominates
and the k neighbours are the nearest tokens along the tangent; for an incoherent (texture / flat) query the
prior is switched off and E_i becomes its k nearest tokens (isotropic local attention), so a noisy
orientation can never cut a token off. Attention for a sparse head is masked to E_i; in "soft" mode a learned
per-head scale gamma_h * s_ij is also added to the logits on the edges. `dense_heads` heads per layer
ignore the graph (the dense fallback head of the design).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
PRIORS = ("ridge", "gradient")


def _gauss_blur(g, sigma):
    r = max(1, int(round(3 * sigma)))
    t = torch.arange(-r, r + 1, dtype=g.dtype, device=g.device)
    k = torch.exp(-t ** 2 / (2 * sigma ** 2)); k = k / k.sum()
    g = F.conv2d(F.pad(g, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(g, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))


def _tensor_to_geometry(txx, txy, tyy, eps):
    """Token tensor components (B, N) -> tangent angle in [0, pi) and coherence in [0, 1]."""
    theta = torch.remainder(0.5 * torch.atan2(2 * txy, txx - tyy) + math.pi / 2, math.pi)
    tr = txx + tyy
    coh = (torch.sqrt((txx - tyy) ** 2 + 4 * txy ** 2) / (tr + eps * tr.mean(1, keepdim=True) + 1e-12)) ** 2
    return theta, coh.clamp(0, 1), tr


@torch.no_grad()
def ridge_geometry(x, grid, sigma=2.0, eps=0.05):
    """Ridge prior -> theta (B, N), gate c (B, N) in [0, 1], coherence (B, N), ridge energy (B, N)."""
    with torch.autocast(x.device.type, enabled=False):
        g = _gauss_blur(x.float().mean(1, keepdim=True), sigma)
        gp = F.pad(g, (1, 1, 1, 1), mode="replicate")
        c = gp[..., 1:-1, 1:-1]
        ixx = gp[..., 1:-1, 2:] - 2 * c + gp[..., 1:-1, :-2]
        iyy = gp[..., 2:, 1:-1] - 2 * c + gp[..., :-2, 1:-1]
        ixy = 0.25 * (gp[..., 2:, 2:] - gp[..., 2:, :-2] - gp[..., :-2, 2:] + gp[..., :-2, :-2])
        half = torch.sqrt(((ixx - iyy) / 2) ** 2 + ixy ** 2)
        lmax, lmin = (ixx + iyy) / 2 + half, (ixx + iyy) / 2 - half
        r = torch.relu(lmax - lmin.abs())                                    # dark-line strength
        psi = 0.5 * torch.atan2(2 * ixy, ixx - iyy)                          # normal (direction of l_max)
        T = F.adaptive_avg_pool2d(torch.cat([r * torch.cos(psi) ** 2, r * torch.cos(psi) * torch.sin(psi),
                                             r * torch.sin(psi) ** 2], 1), grid).flatten(2)
        theta, coh, e = _tensor_to_geometry(*T.unbind(1), eps)
        gate = coh * e / (e.amax(1, keepdim=True) + 1e-12)
    return theta, gate.clamp(0, 1), coh, e


@torch.no_grad()
def token_geometry(x, grid, eps=0.05):
    """x (B, C, H, W) image (any normalisation) -> theta (B, N) tangent angle in [0, pi), coh (B, N) in [0, 1],
    energy (B, N). `eps` is a noise floor relative to the image-mean gradient energy, so faint gradients
    (illumination ramps, flat patches) get low coherence. energy = mean squared gradient (token contrast).
    N = grid * grid, row-major (y, x)."""
    with torch.autocast(x.device.type, enabled=False):
        g = x.float().mean(1, keepdim=True)
        kx = _SOBEL_X.to(g.device)[None, None]
        gx = F.conv2d(F.pad(g, (1, 1, 1, 1), mode="replicate"), kx)
        gy = F.conv2d(F.pad(g, (1, 1, 1, 1), mode="replicate"), kx.transpose(-1, -2))
        J = F.adaptive_avg_pool2d(torch.cat([gx * gx, gx * gy, gy * gy], 1), grid)   # (B, 3, g, g)
        return _tensor_to_geometry(*J.flatten(2).unbind(1), eps)


class SkeletonGraph(nn.Module):
    """Builds the STR-1 edge set E (B, N, N) and edge scores S (B, N, N) for a fixed grid."""

    def __init__(self, grid, k=8, radius=4.0, alpha=30.0, beta=30.0, rho=1.0, mode="soft", coh_gate=True,
                 prior="ridge", sigma=2.0, tau=0.05, log_eps=1e-4):
        super().__init__()
        if mode not in ("hard", "soft"):
            raise ValueError("str_mode must be hard | soft")
        if prior not in PRIORS:
            raise ValueError(f"str_prior must be one of {PRIORS}")
        self.prior, self.sigma = prior, sigma
        self.grid, self.k, self.radius, self.rho, self.mode, self.coh_gate = grid, k, radius, rho, mode, coh_gate
        self.cos_a, self.cos_b = math.cos(math.radians(alpha)), math.cos(math.radians(beta))
        self.tau, self.log_eps = tau, log_eps
        ys, xs = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
        pos = torch.stack([xs.flatten(), ys.flatten()], 1).float()            # (N, 2) = (x, y), image axes
        disp = pos[None] - pos[:, None]                                       # (N, N, 2): p_j - p_i
        dist = disp.norm(dim=-1)
        self.register_buffer("dist", dist, persistent=False)
        self.register_buffer("phi", torch.atan2(disp[..., 1], disp[..., 0]), persistent=False)
        self.register_buffer("cand", (dist > 0) & (dist <= radius), persistent=False)

    @torch.no_grad()
    def forward(self, x):
        """x (B, C, H, W) -> dict(edges (B, N, N) bool incl. self-loops, score (B, N, N) float, 0 off-edge,
        theta (B, N), coh (B, N) = the gate c_i used in the score)."""
        if self.prior == "ridge":
            theta, coh, _, _ = ridge_geometry(x, self.grid, self.sigma)
        else:
            theta, coh, _ = token_geometry(x, self.grid)
        with torch.autocast(x.device.type, enabled=False):
            a = torch.cos(self.phi[None] - theta[:, :, None]).abs()             # cone around i's tangent
            o = torch.cos(theta[:, :, None] - theta[:, None, :]).abs()          # orientation agreement
            if self.mode == "hard":
                g = ((a >= self.cos_a) & (o >= self.cos_b)).float()
            else:
                g = torch.sigmoid((a - self.cos_a) / self.tau) * torch.sigmoid((o - self.cos_b) / self.tau)
            c = coh[:, :, None] if self.coh_gate else 1.0
            s = c * torch.log(g + self.log_eps) - self.dist[None] / self.rho
            s = s.masked_fill(~self.cand[None], float("-inf"))
            k = min(self.k, int(self.cand.sum(1).max()))
            val, idx = s.topk(k, dim=-1)
            edges = torch.zeros_like(s, dtype=torch.bool).scatter_(-1, idx, torch.isfinite(val))
            edges |= torch.eye(s.shape[-1], dtype=torch.bool, device=s.device)[None]
            score = torch.where(edges, s, torch.zeros_like(s)).nan_to_num(0.0, 0.0, 0.0)
            score.diagonal(dim1=-2, dim2=-1).zero_()
        return dict(edges=edges, score=score, theta=theta, coh=coh)


def with_cls(geo):
    """Prepend a [CLS] token: CLS attends to everything and every token may attend to CLS."""
    e, s = geo["edges"], geo["score"]
    B, N, _ = e.shape
    E = torch.ones(B, N + 1, N + 1, dtype=torch.bool, device=e.device)
    E[:, 1:, 1:] = e
    S = torch.zeros(B, N + 1, N + 1, dtype=s.dtype, device=s.device)
    S[:, 1:, 1:] = s
    return dict(geo, edges=E, score=S)


def str_attn_mask(geo, heads, dense_heads, gamma=None, extra=None, dtype=torch.float32):
    """-> additive attention mask (B, heads, N, N). The first `dense_heads` heads see all tokens; the others
    are restricted to the skeleton graph (-inf off-edge) and, if `gamma` (heads - dense_heads,) is given,
    get gamma_h * score on the edges. `extra` (heads, N, N) is added to every head (e.g. relative position bias)."""
    E, S = geo["edges"], geo["score"]
    B, N, _ = E.shape
    hs = heads - dense_heads
    sparse = (gamma.view(1, hs, 1, 1).float() * S[:, None]) if gamma is not None else S.new_zeros(B, 1, N, N)
    sparse = sparse.expand(B, hs, N, N).masked_fill(~E[:, None], float("-inf"))
    mask = torch.cat([S.new_zeros(B, dense_heads, N, N), sparse], 1) if dense_heads else sparse
    if extra is not None:
        mask = mask + extra[None].float()
    return mask.to(dtype)


class GeoAttention(nn.Module):
    """Multi-head self-attention with an optional STR-1 mask. Used by the ViT and the hybrid.
    attention: "mhsa" (dense) | "str1". rel_grid: if set, adds a learned 2-D relative position bias (hybrid)."""

    def __init__(self, dim, heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0, attention="mhsa", dense_heads=1,
                 soft=True, rel_grid=None):
        super().__init__()
        self.heads, self.scale = heads, (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop, self.proj_drop = attn_drop, nn.Dropout(proj_drop)
        self.attention = attention
        self.dense_heads = min(dense_heads, heads) if attention == "str1" else heads
        self.gamma = (nn.Parameter(torch.ones(heads - self.dense_heads))
                      if attention == "str1" and soft and heads > self.dense_heads else None)
        self.rel_grid = rel_grid
        if rel_grid:
            g = rel_grid
            self.rel_bias = nn.Parameter(torch.zeros(heads, (2 * g - 1) ** 2))
            nn.init.trunc_normal_(self.rel_bias, std=0.02)
            ys, xs = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
            c = torch.stack([ys.flatten(), xs.flatten()])
            rel = (c[:, :, None] - c[:, None, :]) + (g - 1)
            self.register_buffer("rel_index", rel[0] * (2 * g - 1) + rel[1], persistent=False)
        self.store_attn = False          # set by gradcam.py: explicit softmax so the map can be read
        self.attn = None

    def forward(self, x, geo=None):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        extra = self.rel_bias[:, self.rel_index] if self.rel_grid else None
        if self.attention == "str1" and geo is not None and self.dense_heads < self.heads:
            mask = str_attn_mask(geo, self.heads, self.dense_heads, self.gamma, extra, q.dtype)
        else:
            mask = extra[None].to(q.dtype) if extra is not None else None
        if self.store_attn:
            logits = (q * self.scale) @ k.transpose(-2, -1)
            a = (logits + mask if mask is not None else logits).softmax(-1)
            self.attn = a
            x = a @ v
        else:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                               dropout_p=self.attn_drop if self.training else 0.0)
        return self.proj_drop(self.proj(x.transpose(1, 2).reshape(B, N, C)))
