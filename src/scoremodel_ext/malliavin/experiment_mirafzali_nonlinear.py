"""
Mirafzali reproduction — Phase B: Nonlinear SDE experiments
============================================================
Full pipeline:
  dataset → nonlinear SDE (Malliavin forward) → teacher → MLP training →
  reverse Euler → metrics → output

Teachers : raw (Skorokhod), binned, nw, knn_nw
Datasets : 8gmm, checkerboard, swissroll

Output layout
-------------
results/mirafzali_nonlinear/{dataset}/{method}/
  teacher_field.png
  reverse_samples.png
  metrics.json
  model.pt

Run
---
python -m scoremodel_ext.malliavin.experiment_mirafzali_nonlinear
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

try:
    from .datasets_2d import get_sampler, sample_8gmm
    from .models import TimeScoreMLP2D, train_mirafzali_skorokhod_net
    from .sde_2d import bin_teacher_2d, nw_teacher_2d, knn_nw_teacher_2d
    from .sde_nonlinear import (
        NonlinearSDEConfig,
        simulate_malliavin_nl,
        simulate_forward_nl,
        reverse_euler_nl,
        sample_stationary_nl,
    )
    from .experiment_mirafzali import (
        _mmd_rbf,
        _sliced_wasserstein,
        _mode_coverage_8gmm,
        train_score_mlp,
        _plot_scatter,
        _plot_teacher_field,
    )
except ImportError:
    from datasets_2d import get_sampler, sample_8gmm
    from models import TimeScoreMLP2D, train_mirafzali_skorokhod_net
    from sde_2d import bin_teacher_2d, nw_teacher_2d, knn_nw_teacher_2d
    from sde_nonlinear import (
        NonlinearSDEConfig,
        simulate_malliavin_nl,
        simulate_forward_nl,
        reverse_euler_nl,
    )
    from experiment_mirafzali import (
        _mmd_rbf,
        _sliced_wasserstein,
        _mode_coverage_8gmm,
        train_score_mlp,
        _plot_scatter,
        _plot_teacher_field,
    )

plt.style.use("seaborn-v0_8")

# Default SDE config (Mirafzali Appendix C)
DEFAULT_NL_CFG = NonlinearSDEConfig(
    k=1.0, sigma=1.0, a=0.0,
    beta_min=1.0, beta_max=25.0, T=1.0,
)

TIMES: List[float] = [0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00]

# n_steps scales with T so that dt ≈ 0.004 throughout
_N_STEPS_PER_UNIT = 250   # steps for T = 1.0


def _n_steps_for(T: float, n_steps_per_unit: int = _N_STEPS_PER_UNIT) -> int:
    return max(10, round(T * n_steps_per_unit))


# ──────────────────────────────────────────────────────────────────────────────
# Teacher helpers
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _binned_score_at_points(
    X_T: torch.Tensor,
    H: torch.Tensor,
    query_x: torch.Tensor,
    n_bins: int = 60,
    q_low: float = 0.005,
    q_high: float = 0.995,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    For each point in *query_x*, return the bin-averaged score from (X_T, H).

    The 2-D histogram is built from the [q_low, q_high] quantile range of X_T.
    Query points outside that range are clamped to the nearest boundary bin.

    Returns
    -------
    query_x : (m, 2) — same tensor passed in
    sc      : (m, 2) — bin-averaged H at each query point
    cc      : (m,)   — uniform weights (ones)
    """
    device = X_T.device
    q = torch.tensor([q_low, q_high], device=device)

    x_min, x_max = torch.quantile(X_T[:, 0], q)
    y_min, y_max = torch.quantile(X_T[:, 1], q)

    x_edges = torch.linspace(x_min.item(), x_max.item(), n_bins + 1, device=device)
    y_edges = torch.linspace(y_min.item(), y_max.item(), n_bins + 1, device=device)
    n_cells = n_bins * n_bins

    # Bin ALL X_T to accumulate per-bin sums
    x0_all = X_T[:, 0].contiguous()
    x1_all = X_T[:, 1].contiguous()
    ix_all = (torch.bucketize(x0_all, x_edges) - 1).clamp(0, n_bins - 1)
    iy_all = (torch.bucketize(x1_all, y_edges) - 1).clamp(0, n_bins - 1)
    flat_all = ix_all * n_bins + iy_all

    counts = torch.bincount(flat_all, minlength=n_cells).float()
    sum0   = torch.bincount(flat_all, weights=H[:, 0].contiguous(), minlength=n_cells)
    sum1   = torch.bincount(flat_all, weights=H[:, 1].contiguous(), minlength=n_cells)
    avg_h0 = sum0 / counts.clamp_min(1.0)
    avg_h1 = sum1 / counts.clamp_min(1.0)

    # Look up bin average for each query point
    ix_q   = (torch.bucketize(query_x[:, 0].contiguous(), x_edges) - 1).clamp(0, n_bins - 1)
    iy_q   = (torch.bucketize(query_x[:, 1].contiguous(), y_edges) - 1).clamp(0, n_bins - 1)
    flat_q = ix_q * n_bins + iy_q

    sc = torch.stack([avg_h0[flat_q], avg_h1[flat_q]], dim=1)
    cc = torch.ones(query_x.shape[0], device=device)
    return query_x, sc, cc


# ──────────────────────────────────────────────────────────────────────────────
# Teacher methods
# ──────────────────────────────────────────────────────────────────────────────

def apply_teacher_nl(
    method: str,
    X_T: torch.Tensor,
    H: torch.Tensor,
    n_raw: int = 20_000,
    n_bins: int = 60,
    min_count: int = 20,
    knn_k: int = 256,
    bandwidth_scale: float = 1.0,
    teacher_eval_points: str = "raw_points",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply a teacher to (X_T, H) pairs from the nonlinear SDE simulation.

    Parameters
    ----------
    method : "raw" | "binned" | "nw" | "knn_nw"
    X_T    : (n, 2) terminal forward samples
    H      : (n, 2) Malliavin score weights
    teacher_eval_points : "raw_points" | "grid_centers"
        "raw_points"   — all methods share the same *n_raw* subsampled X_T
                         query points; only the score estimation strategy
                         differs.  This ensures a fair, equal-count comparison.
        "grid_centers" — binned/nw/knn_nw use 2-D histogram bin centres as
                         query points (original behaviour); raw uses subsampled
                         X_T.

    Returns
    -------
    pts    : (m, 2) query positions
    scores : (m, 2) estimated scores
    counts : (m,)   per-point weights
    """
    if teacher_eval_points not in ("raw_points", "grid_centers"):
        raise ValueError(
            f"teacher_eval_points must be 'raw_points' or 'grid_centers', "
            f"got {teacher_eval_points!r}"
        )

    # ── raw_points mode: all methods share the same n_raw query positions ──
    if teacher_eval_points == "raw_points":
        n   = X_T.shape[0]
        idx = torch.randperm(n, device=X_T.device)[:min(n_raw, n)]
        query_x = X_T[idx]

        if method == "raw":
            return query_x, H[idx], torch.ones(idx.shape[0], device=X_T.device)

        if method == "binned":
            return _binned_score_at_points(X_T, H, query_x, n_bins=n_bins)

        if method == "nw":
            sc = nw_teacher_2d(X_T, H, query_x)
            return query_x, sc, torch.ones(idx.shape[0], device=X_T.device)

        if method == "knn_nw":
            sc = knn_nw_teacher_2d(
                X_T, H, query_x, k=knn_k, bandwidth_scale=bandwidth_scale,
            )
            return query_x, sc, torch.ones(idx.shape[0], device=X_T.device)

        if method == "mirafzali":
            # Algorithm 6: use ALL n_paths — no subsampling, equal weights.
            # This is the faithful Mirafzali baseline: N_θ trained on the
            # full dataset {(X_t, t, δ_t(u_t))} with plain MSE.
            cc = torch.ones(X_T.shape[0], device=X_T.device)
            return X_T, H, cc

        raise ValueError(
            f"Unknown teacher method {method!r}. "
            f"Choose from 'raw', 'binned', 'nw', 'knn_nw', 'mirafzali'."
        )

    # ── grid_centers mode: existing behaviour ─────────────────────────────
    if method == "raw":
        n   = X_T.shape[0]
        idx = torch.randperm(n, device=X_T.device)[:min(n_raw, n)]
        pts = X_T[idx]
        sc  = H[idx]
        cc  = torch.ones(pts.shape[0], device=X_T.device)
        return pts, sc, cc

    if method == "binned":
        return bin_teacher_2d(X_T, H, n_bins=n_bins, min_count=min_count)

    # NW and kNN-NW use the binned grid as query positions
    if method in ("nw", "knn_nw"):
        pts, _, cc = bin_teacher_2d(X_T, H, n_bins=n_bins, min_count=min_count)
        if pts.shape[0] == 0:
            return pts, pts.clone(), cc
        if method == "nw":
            sc = nw_teacher_2d(X_T, H, pts)
        else:
            sc = knn_nw_teacher_2d(
                X_T, H, pts, k=knn_k, bandwidth_scale=bandwidth_scale,
            )
        return pts, sc, cc

    if method == "mirafzali":
        # Algorithm 6: use ALL n_paths — no subsampling, equal weights.
        cc = torch.ones(X_T.shape[0], device=X_T.device)
        return X_T, H, cc

    raise ValueError(
        f"Unknown teacher method {method!r}. "
        f"Choose from 'raw', 'binned', 'nw', 'knn_nw', 'mirafzali'."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Simulation cache
# ──────────────────────────────────────────────────────────────────────────────

def simulate_all_times_nl(
    times: Sequence[float],
    dataset_name: str,
    cfg: NonlinearSDEConfig = DEFAULT_NL_CFG,
    n_paths: int = 50_000,
    n_steps_per_unit: int = _N_STEPS_PER_UNIT,
    gamma_reg: float = 1e-3,
    device: str = "cpu",
    correction: str = "approx",
) -> List[Tuple[float, torch.Tensor, torch.Tensor]]:
    """
    For each T in *times*, sample X_0 from the dataset and simulate to X_T
    with Malliavin score weights via the nonlinear SDE.

    n_steps is proportional to T so that dt ≈ const throughout.

    Returns
    -------
    List of (T, X_T, H) tuples, all tensors on *device*.
    """
    sampler = get_sampler(dataset_name)
    cache = []

    for T in times:
        result = sampler(n_paths, device=device)
        X0     = result[0] if isinstance(result, tuple) else result
        n_steps = _n_steps_for(T, n_steps_per_unit)

        print(f"  Simulating T={T:.2f}  n_steps={n_steps} …", flush=True)
        X_T, H = simulate_malliavin_nl(
            X0, T, cfg, n_steps=n_steps, gamma_reg=gamma_reg,
            correction=correction,
        )
        cache.append((T, X_T, H))

    return cache


# ──────────────────────────────────────────────────────────────────────────────
# Dataset builder
# ──────────────────────────────────────────────────────────────────────────────

def build_training_dataset_nl(
    sim_cache: List[Tuple[float, torch.Tensor, torch.Tensor]],
    method: str,
    n_raw: int = 20_000,
    n_bins: int = 60,
    min_count: int = 20,
    knn_k: int = 256,
    bandwidth_scale: float = 1.0,
    teacher_eval_points: str = "raw_points",
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Apply the teacher to each cached (T, X_T, H) and concatenate across time.

    Returns (t, x, s, c) or None if every time step yields zero valid points.
    """
    t_list, x_list, s_list, c_list = [], [], [], []
    device = sim_cache[0][1].device

    for T, X_T, H in sim_cache:
        pts, sc, cc = apply_teacher_nl(
            method, X_T, H,
            n_raw=n_raw, n_bins=n_bins, min_count=min_count,
            knn_k=knn_k, bandwidth_scale=bandwidth_scale,
            teacher_eval_points=teacher_eval_points,
        )
        if pts.shape[0] == 0:
            continue
        t_list.append(torch.full((pts.shape[0],), T, device=device))
        x_list.append(pts)
        s_list.append(sc)
        c_list.append(cc)

    if not t_list:
        return None
    return (
        torch.cat(t_list),
        torch.cat(x_list),
        torch.cat(s_list),
        torch.cat(c_list),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics_nl(
    samples_np: np.ndarray,
    nan_rate: float,
    dataset_name: str,
    centers_np: Optional[np.ndarray] = None,
    n_ref: int = 10_000,
    rng=None,
) -> dict:
    """
    Compute MMD, sliced Wasserstein, and (for 8GMM) mode coverage.

    Reference samples are drawn fresh from the dataset distribution.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    sampler  = get_sampler(dataset_name)
    result_r = sampler(n_ref, device="cpu")
    ref_np   = (result_r[0] if isinstance(result_r, tuple) else result_r).numpy()

    metrics: dict = {
        "nan_rate":           float(nan_rate),
        "n_samples":          int(len(samples_np)),
        "mmd_rbf":            _mmd_rbf(samples_np, ref_np, rng=rng),
        "sliced_wasserstein": _sliced_wasserstein(samples_np, ref_np, rng=rng),
    }

    if dataset_name == "8gmm" and centers_np is not None:
        metrics["mode_coverage"] = _mode_coverage_8gmm(samples_np, centers_np)

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Residual correction helpers
# ──────────────────────────────────────────────────────────────────────────────

_RESIDUAL_METHODS = (
    "mirafzali_residual_binned",
    "mirafzali_residual_nw",
    "mirafzali_residual_knn_nw",
)


@torch.no_grad()
def compute_residuals_nl(
    model,
    t_tr: torch.Tensor,
    x_tr: torch.Tensor,
    H_tr: torch.Tensor,
    device: str = "cpu",
    batch_size: int = 4_096,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute per-sample residuals  r_i = H_i − model(t_i, x_i).

    Parameters
    ----------
    model     : trained score model; must support (t: Tensor(b,), x: Tensor(b,2))
    t_tr      : (n,)   training time labels
    x_tr      : (n, 2) training positions
    H_tr      : (n, 2) Malliavin score weights (targets)
    device    : device to run model inference on
    batch_size: mini-batch size for model inference

    Returns
    -------
    r_tr        : (n, 2) CPU tensor of residuals
    diagnostics : dict with var_H, var_residual, mean_residual_norm
    """
    model.eval()
    model = model.to(device)
    preds = []
    n = t_tr.shape[0]
    for i in range(0, n, batch_size):
        t_b = t_tr[i : i + batch_size].to(device)
        x_b = x_tr[i : i + batch_size].to(device)
        preds.append(model(t_b, x_b).cpu())
    pred_all = torch.cat(preds)
    r_tr = H_tr.cpu() - pred_all

    diagnostics = {
        "var_H":              float(H_tr.var().item()),
        "var_residual":       float(r_tr.var().item()),
        "mean_residual_norm": float(r_tr.norm(dim=1).mean().item()),
    }
    return r_tr, diagnostics


@torch.no_grad()
def _precompute_residual_bins(
    X_T: torch.Tensor,
    R: torch.Tensor,
    n_bins: int,
) -> dict:
    """Build a 2-D histogram of bin-averaged residuals (CPU tensors)."""
    q = torch.tensor([0.005, 0.995])
    x_min, x_max = torch.quantile(X_T[:, 0], q)
    y_min, y_max = torch.quantile(X_T[:, 1], q)

    x_edges = torch.linspace(x_min.item(), x_max.item(), n_bins + 1)
    y_edges = torch.linspace(y_min.item(), y_max.item(), n_bins + 1)
    n_cells = n_bins * n_bins

    ix = (torch.bucketize(X_T[:, 0].contiguous(), x_edges) - 1).clamp(0, n_bins - 1)
    iy = (torch.bucketize(X_T[:, 1].contiguous(), y_edges) - 1).clamp(0, n_bins - 1)
    flat = ix * n_bins + iy

    counts = torch.bincount(flat, minlength=n_cells).float()
    avg_r0 = torch.bincount(flat, weights=R[:, 0].contiguous(), minlength=n_cells) / counts.clamp_min(1.0)
    avg_r1 = torch.bincount(flat, weights=R[:, 1].contiguous(), minlength=n_cells) / counts.clamp_min(1.0)

    return {
        "x_edges": x_edges,
        "y_edges": y_edges,
        "avg_r0":  avg_r0,
        "avg_r1":  avg_r1,
        "n_bins":  n_bins,
    }


@torch.no_grad()
def _apply_bin_residual(query_x: torch.Tensor, bt: dict) -> torch.Tensor:
    """Look up bin-averaged residual correction at *query_x* positions."""
    n_bins  = bt["n_bins"]
    x_edges = bt["x_edges"]
    y_edges = bt["y_edges"]
    ix = (torch.bucketize(query_x[:, 0].contiguous(), x_edges) - 1).clamp(0, n_bins - 1)
    iy = (torch.bucketize(query_x[:, 1].contiguous(), y_edges) - 1).clamp(0, n_bins - 1)
    flat = ix * n_bins + iy
    return torch.stack([bt["avg_r0"][flat], bt["avg_r1"][flat]], dim=1)


@torch.no_grad()
def _nw_residual(
    query_x: torch.Tensor,
    X_ref: torch.Tensor,
    R_ref: torch.Tensor,
    bandwidth_scale: float = 1.0,
    batch_size: int = 512,
) -> torch.Tensor:
    """
    Nadaraya-Watson estimate of E[R | X = query_x] from reference set (X_ref, R_ref).

    Bandwidth = bandwidth_scale × Silverman's rule (d=2).
    """
    n_ref = X_ref.shape[0]
    sigma_mean = float(X_ref.std(dim=0).mean().clamp_min(1e-3))
    h = bandwidth_scale * sigma_mean * float(n_ref) ** (-1.0 / 6.0)

    nq  = query_x.shape[0]
    out = torch.zeros(nq, 2, device=query_x.device, dtype=query_x.dtype)
    for i in range(0, nq, batch_size):
        xq   = query_x[i : i + batch_size]
        diff = (xq[:, None, :] - X_ref[None, :, :]) / h
        kw   = torch.exp(-0.5 * (diff * diff).sum(-1))
        denom = kw.sum(-1, keepdim=True).clamp_min(1e-12)
        out[i : i + batch_size] = (kw[:, :, None] * R_ref[None, :, :]).sum(1) / denom
    return out


@torch.no_grad()
def _knn_nw_residual(
    query_x: torch.Tensor,
    X_ref: torch.Tensor,
    R_ref: torch.Tensor,
    k: int = 256,
    bandwidth_scale: float = 1.0,
    batch_size: int = 256,
) -> torch.Tensor:
    """
    kNN-adaptive Nadaraya-Watson estimate of E[R | X = query_x].

    Bandwidth per query = bandwidth_scale × distance to k-th nearest reference point.
    """
    return knn_nw_teacher_2d(
        X_ref, R_ref, query_x,
        k=k, bandwidth_scale=bandwidth_scale, batch_size=batch_size,
    )


class ResidualCorrectionModel:
    """
    Wraps a trained base score model with a per-time-step nonparametric
    residual correction:

        score(t, x) = base_model(t, x) + alpha * r̂(t, x)

    r̂(t, x) ≈ E[r | X_t = x] is estimated from training residuals stored at
    each discrete time step.  At inference the nearest stored time is used.

    This is a plain Python callable (not ``nn.Module``), with ``eval()`` and
    ``to()`` pass-throughs for compatibility with ``reverse_euler_nl``.

    Parameters
    ----------
    base_model          : trained score model, (t: Tensor(n,), x: Tensor(n,2)) → Tensor(n,2)
    times               : list of discrete training times (sorted ascending)
    X_T_list            : training positions per time step (CPU tensors)
    R_list              : training residuals per time step (CPU tensors)
    mode                : ``'binned'`` | ``'nw'`` | ``'knn_nw'``
    alpha               : residual shrinkage weight (1.0 = full correction)
    n_bins              : spatial bins per axis (for 'binned' mode)
    nw_n_ref            : max reference points per time step (for 'nw'/'knn_nw')
    nw_bandwidth_scale  : Silverman bandwidth multiplier (for 'nw' mode)
    knn_k               : number of nearest neighbours (for 'knn_nw' mode)
    knn_bandwidth_scale : local bandwidth multiplier (for 'knn_nw' mode)
    nw_batch            : query batch size for NW/knn_nw inference
    """

    def __init__(
        self,
        base_model,
        times: List[float],
        X_T_list: List[torch.Tensor],
        R_list: List[torch.Tensor],
        mode: str = "binned",
        alpha: float = 1.0,
        n_bins: int = 60,
        nw_n_ref: int = 5_000,
        nw_bandwidth_scale: float = 1.0,
        knn_k: int = 256,
        knn_bandwidth_scale: float = 1.0,
        nw_batch: int = 512,
    ) -> None:
        if mode not in ("binned", "nw", "knn_nw"):
            raise ValueError(f"ResidualCorrectionModel: unknown mode {mode!r}.")
        self.base_model         = base_model
        self.times              = list(times)
        self.mode               = mode
        self.alpha              = alpha
        self.nw_batch           = nw_batch
        self.nw_bandwidth_scale = nw_bandwidth_scale
        self.knn_k              = knn_k
        self.knn_bandwidth_scale = knn_bandwidth_scale

        if mode == "binned":
            # Precompute bin tables on CPU; tensors transferred to device in forward().
            self._bin_tables = [
                _precompute_residual_bins(Xt.cpu(), R.cpu(), n_bins)
                for Xt, R in zip(X_T_list, R_list)
            ]
            self._X_T_list = None
            self._R_list   = None
        else:  # nw or knn_nw
            self._bin_tables = None
            # Subsample reference set to nw_n_ref per time step.
            self._X_T_list = []
            self._R_list   = []
            for Xt, R in zip(X_T_list, R_list):
                n   = Xt.shape[0]
                idx = torch.randperm(n)[:min(n, nw_n_ref)]
                self._X_T_list.append(Xt[idx].cpu())
                self._R_list.append(R[idx].cpu())

    # ── pass-throughs for nn.Module-style API ──────────────────────────────
    def eval(self):
        self.base_model.eval()
        return self

    def to(self, device):
        self.base_model = self.base_model.to(device)
        return self

    # ── core ──────────────────────────────────────────────────────────────
    def _nearest_idx(self, t_val: float) -> int:
        return min(range(len(self.times)), key=lambda i: abs(self.times[i] - t_val))

    def __call__(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        base_score = self.base_model(t, x)
        if self.alpha == 0.0:
            return base_score
        device = x.device
        t_val  = float(t.reshape(-1)[0].item())
        idx    = self._nearest_idx(t_val)

        if self.mode == "binned":
            bt = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in self._bin_tables[idx].items()
            }
            r_hat = _apply_bin_residual(x, bt)
        elif self.mode == "nw":
            X_ref = self._X_T_list[idx].to(device)
            R_ref = self._R_list[idx].to(device)
            r_hat = _nw_residual(x, X_ref, R_ref,
                                 bandwidth_scale=self.nw_bandwidth_scale,
                                 batch_size=self.nw_batch)
        else:  # knn_nw
            X_ref = self._X_T_list[idx].to(device)
            R_ref = self._R_list[idx].to(device)
            r_hat = _knn_nw_residual(x, X_ref, R_ref,
                                     k=self.knn_k,
                                     bandwidth_scale=self.knn_bandwidth_scale,
                                     batch_size=self.nw_batch)

        return base_score + self.alpha * r_hat


def _plot_residual_field(
    corrected_model: ResidualCorrectionModel,
    T: float,
    device: str,
    outpath: Path,
    n_grid: int = 30,
) -> None:
    """
    Save a quiver plot of the base, corrected and residual score fields at
    time *T* over a regular 2-D spatial grid.
    """
    # Build grid from [-4, 4]² (covers swissroll / 8gmm / checkerboard)
    xs = torch.linspace(-4.0, 4.0, n_grid, device=device)
    ys = torch.linspace(-4.0, 4.0, n_grid, device=device)
    gX, gY = torch.meshgrid(xs, ys, indexing="ij")
    grid = torch.stack([gX.reshape(-1), gY.reshape(-1)], dim=1)  # (n_grid², 2)
    t_vec = torch.full((grid.shape[0],), T, device=device)

    with torch.no_grad():
        base      = corrected_model.base_model(t_vec, grid).cpu().numpy()
        corrected = corrected_model(t_vec, grid).cpu().numpy()
    residual  = corrected - base
    xy        = grid.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    scale = max(float(np.abs(corrected).max()), 1.0) * 5.0
    for ax, field, title in zip(
        axes,
        [base, corrected, residual],
        ["Base model", "Corrected model", "Residual correction"],
    ):
        ax.quiver(
            xy[:, 0], xy[:, 1], field[:, 0], field[:, 1],
            angles="xy", scale_units="xy", scale=scale, alpha=0.75,
        )
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)
    fig.suptitle(f"Residual correction field  T={T:.2f}", fontsize=13)
    fig.tight_layout()
    fig.savefig(outpath, dpi=100)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Per-run pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_experiment_nl(
    dataset: str,
    method: str,
    cfg: NonlinearSDEConfig = DEFAULT_NL_CFG,
    times: List[float] = TIMES,
    n_paths: int = 50_000,
    n_steps_per_unit: int = _N_STEPS_PER_UNIT,
    gamma_reg: float = 1e-3,
    n_epochs: int = 8_000,
    batch_size: int = 4_096,
    lr: float = 2e-4,
    n_bins: int = 60,
    min_count: int = 20,
    knn_k: int = 256,
    n_raw: int = 20_000,
    n_samples_reverse: int = 20_000,
    outbase: str = "results/mirafzali_nonlinear",
    device: Optional[str] = None,
    seed: int = 0,
    reverse_init: str = "stationary",
    mirafzali_mode: bool = True,
    teacher_eval_points: str = "raw_points",
    correction: str = "approx",
    # ── residual correction hyper-parameters ────────────────────────────
    residual_alpha: float = 1.0,
    nw_bandwidth_scale: float = 1.0,
    residual_knn_k: int = 256,
    residual_knn_bandwidth_scale: float = 1.0,
    # ── internal: skip simulation (sweep reuse) ─────────────────────────
    _sim_cache: Optional[list] = None,
    _outdir_override: Optional[str] = None,
    # ── MirafzaliSkorokhodNet architecture ─────────────────────────────────
    hidden: int = 512,
    n_blocks: int = 6,
    num_frequencies: int = 16,
    weight_decay: float = 1e-5,
) -> dict:
    """
    Full pipeline for one (dataset, method) combination with the nonlinear SDE.

    Returns metrics dict.
    """
    is_residual  = method in _RESIDUAL_METHODS
    _base_method = "mirafzali" if is_residual else method
    _residual_mode = (
        "binned"  if method == "mirafzali_residual_binned"  else
        "nw"      if method == "mirafzali_residual_nw"      else
        "knn_nw"  if method == "mirafzali_residual_knn_nw"  else
        None
    )

    if mirafzali_mode:
        n_paths = 25_000
        n_epochs = 2500
        batch_size = 2048
        lr = 3e-4
        n_steps_rev = 250

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    outdir = Path(_outdir_override) if _outdir_override else Path(outbase) / dataset / method
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  dataset={dataset}  method={method}  device={device}")
    if is_residual:
        print(f"  alpha={residual_alpha}  nw_bw={nw_bandwidth_scale}  "
              f"knn_k={residual_knn_k}  knn_bw={residual_knn_bandwidth_scale}")
    print(f"{'='*60}", flush=True)

    # ── forward simulation (skipped when pre-computed cache is provided) ──
    if _sim_cache is not None:
        sim_cache = _sim_cache
        print("  Using pre-computed simulation cache.", flush=True)
    else:
        print("Simulating forward paths (Malliavin) …", flush=True)
        sim_cache = simulate_all_times_nl(
            times, dataset, cfg,
            n_paths=n_paths, n_steps_per_unit=n_steps_per_unit,
            gamma_reg=gamma_reg, device=device,
            correction=correction,
        )

    # ── build training dataset ─────────────────────────────────────────────
    # Residual methods train the base mirafzali model first.
    print("Building teacher dataset …", flush=True)
    dataset_tuple = build_training_dataset_nl(
        sim_cache,
        _base_method,
        n_raw=n_raw,
        n_bins=n_bins,
        min_count=min_count,
        knn_k=knn_k,
        teacher_eval_points=teacher_eval_points,
    )
    if dataset_tuple is None:
        print("  ERROR: no valid teacher points found, aborting.", flush=True)
        return {"error": "no_valid_teacher_points"}

    t_tr, x_tr, s_tr, c_tr = dataset_tuple
    print(f"  training points: {t_tr.shape[0]:,}", flush=True)

    # Plot teacher field at T_max
    T_last = times[-1]
    t_mask = (t_tr == T_last)
    pts_np = x_tr[t_mask].cpu().numpy()
    sc_np  = s_tr[t_mask].cpu().numpy()
    if len(pts_np) > 0:
        _plot_teacher_field(
            pts_np, sc_np, T_last,
            f"Teacher field ({method}), T={T_last:.2f}",
            outdir / "teacher_field.png",
        )

    # ── train model ────────────────────────────────────────────────────────
    if _base_method == "mirafzali":
        # Algorithm 6: Fourier-feature residual MLP, AdamW, cosine LR,
        # input/target normalisation, plain MSE (no per-point weights).
        print("Training MirafzaliSkorokhodNet …", flush=True)
        model = train_mirafzali_skorokhod_net(
            t_tr, x_tr, s_tr,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            hidden=hidden,
            n_blocks=n_blocks,
            num_frequencies=num_frequencies,
            device=device,
        )
    else:
        print("Training score MLP …", flush=True)
        model = train_score_mlp(
            t_tr, x_tr, s_tr, c_tr,
            n_epochs=n_epochs, batch_size=batch_size, lr=lr, device=device,
        )
    # Save base model (always; residual data is not bundled to keep file small).
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "dataset": dataset,
            "method": method,
            "base_method": _base_method,
            "sde": "nonlinear",
            "reverse_init": reverse_init,
            "cfg": cfg.__dict__ if hasattr(cfg, "__dict__") else str(cfg),
        },
        outdir / "model.pt",
    )

    # ── residual correction (for mirafzali_residual_* methods) ─────────────
    _resid_diags: dict = {}
    if is_residual:
        assert _residual_mode is not None
        print(f"Computing residuals ({_residual_mode}) …", flush=True)
        r_tr, _resid_diags = compute_residuals_nl(
            model, t_tr, x_tr, s_tr, device=device,
        )
        print(
            f"  var_H={_resid_diags['var_H']:.4f}  "
            f"var_r={_resid_diags['var_residual']:.4f}  "
            f"mean‖r‖={_resid_diags['mean_residual_norm']:.4f}",
            flush=True,
        )

        # Group training data by discrete time step.
        _unique_times = sorted({float(t.item()) for t in t_tr.unique()})
        _X_T_by_t, _R_by_t = [], []
        for _Tc in _unique_times:
            _mask = (t_tr.cpu() - _Tc).abs() < 1e-5
            _X_T_by_t.append(x_tr.cpu()[_mask])
            _R_by_t.append(r_tr[_mask])

        print("Building residual correction model …", flush=True)
        model = ResidualCorrectionModel(
            model, _unique_times, _X_T_by_t, _R_by_t,
            mode=_residual_mode,
            alpha=residual_alpha,
            n_bins=n_bins,
            nw_bandwidth_scale=nw_bandwidth_scale,
            knn_k=residual_knn_k,
            knn_bandwidth_scale=residual_knn_bandwidth_scale,
        )

        print("Plotting residual field …", flush=True)
        _plot_residual_field(
            model, times[-1], device, outdir / "residual_field.png",
        )

    # ── reverse sampling ───────────────────────────────────────────────────
    print("Reverse sampling …", flush=True)
    model.eval()
    model = model.to(device)

    # Choose reverse initial distribution.
    #
    # Mirafzali nonlinear sampling starts from the stationary distribution.
    # The old "forward_terminal" option is a reconstruction / consistency test:
    #   X0 ~ data -> forward -> X_T -> reverse -> X0.
    sampler = get_sampler(dataset)
    n_steps_rev = 250

    if reverse_init == "stationary":
        X_T_start = sample_stationary_nl(
            n_samples_reverse,
            cfg,
            dim=2,
            device=device,
            clamp=20.0,
        )
    elif reverse_init == "forward_terminal":
        result = sampler(n_samples_reverse, device=device)
        X0_rev = result[0] if isinstance(result, tuple) else result
        X_T_start = simulate_forward_nl(
            X0_rev,
            cfg.T,
            cfg,
            n_steps=n_steps_rev,
        )
    else:
        raise ValueError(
            f"Unknown reverse_init={reverse_init!r}. "
            "Use 'stationary' or 'forward_terminal'."
        )

    samples = reverse_euler_nl(model, X_T_start, cfg, n_steps=n_steps_rev)

    nan_mask = torch.isnan(samples).any(dim=1)
    nan_rate = float(nan_mask.float().mean().item())
    clean    = samples[~nan_mask].numpy()
    print(f"  {len(clean):,} clean samples  (nan_rate={nan_rate:.3f})", flush=True)

    # ── metrics ────────────────────────────────────────────────────────────
    centers_np = None
    if dataset == "8gmm":
        _, centers_t = sample_8gmm(1, device="cpu")
        centers_np   = centers_t.numpy()

    metrics = compute_metrics_nl(
        clean, nan_rate,
        dataset_name=dataset,
        centers_np=centers_np,
        rng=np.random.default_rng(seed),
    )
    metrics["reverse_init"] = reverse_init
    if is_residual:
        metrics["residual_alpha"]              = residual_alpha
        metrics["nw_bandwidth_scale"]          = nw_bandwidth_scale
        metrics["residual_knn_k"]              = residual_knn_k
        metrics["residual_knn_bandwidth_scale"] = residual_knn_bandwidth_scale
    if _resid_diags:
        metrics.update(_resid_diags)
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(
        f"  mmd={metrics['mmd_rbf']:.5f}  "
        f"sw={metrics['sliced_wasserstein']:.5f}",
        flush=True,
    )

    # ── scatter plot ───────────────────────────────────────────────────────
    result_r = sampler(5_000, device="cpu")
    ref_np   = (result_r[0] if isinstance(result_r, tuple) else result_r).numpy()
    _plot_scatter(
        clean[:5_000],
        f"Reverse samples — nonlinear/{dataset}/{method}",
        outdir / "reverse_samples.png",
        ref_np=ref_np,
    )

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Results table
# ──────────────────────────────────────────────────────────────────────────────

_METHOD_ORDER = [
    "raw", "binned", "nw", "knn_nw",
    "mirafzali",
    "mirafzali_residual_binned",
    "mirafzali_residual_nw",
    "mirafzali_residual_knn_nw",
]


def build_results_table(
    results: dict,
    outbase: Optional[str] = None,
) -> "pd.DataFrame":
    """
    Flatten a nested results dict into a paper-style evaluation table.

    Parameters
    ----------
    results : dict
        Nested dict as returned by ``run_phase_b``:
        ``{dataset: {method: metrics_dict}}``.
    outbase : str or None
        If given, write ``summary.csv`` and ``summary.tex`` under this
        directory.  The directory is created if it does not exist.

    Returns
    -------
    pd.DataFrame
        Columns: dataset, method, mmd, sliced_wasserstein, nan_rate,
        and (for 8gmm rows only) coverage_fraction, mean_nearest_dist.
        Within each dataset block, the best (minimum) value in each
        numeric column is marked with a trailing ``*``.
    """
    base_cols   = ["dataset", "method", "mmd", "sliced_wasserstein", "nan_rate"]
    gmm_cols    = ["coverage_fraction", "mean_nearest_dist"]
    metric_cols = ["mmd", "sliced_wasserstein", "nan_rate"] + gmm_cols

    rows = []
    for dataset in sorted(results.keys()):
        method_dict = results[dataset]
        for method in _METHOD_ORDER:
            if method not in method_dict:
                continue
            m = method_dict[method]
            if "error" in m:
                continue
            row: dict = {
                "dataset":           dataset,
                "method":            method,
                "mmd":               m.get("mmd_rbf"),
                "sliced_wasserstein": m.get("sliced_wasserstein"),
                "nan_rate":          m.get("nan_rate"),
            }
            mc = m.get("mode_coverage", {})
            row["coverage_fraction"] = mc.get("coverage_fraction") if mc else None
            row["mean_nearest_dist"] = mc.get("mean_nearest_dist") if mc else None
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=base_cols + gmm_cols)

    df = pd.DataFrame(rows, columns=base_cols + gmm_cols)

    # Convert metric columns to object dtype so we can append "*" to strings
    # without triggering a dtype incompatibility warning in recent pandas.
    for col in metric_cols:
        df[col] = df[col].astype(object)

    # ── mark best value per dataset ───────────────────────────────────────
    # Lower is better for all metrics; mark minimum with "*".
    # We build a string version of the table for display / LaTeX export.
    str_rows = []
    for dataset, grp in df.groupby("dataset", sort=False):
        for col in metric_cols:
            numeric_vals = pd.to_numeric(grp[col], errors="coerce")
            if numeric_vals.notna().any():
                best_idx = numeric_vals.idxmin()
                df.loc[best_idx, col] = str(df.loc[best_idx, col]) + "*"
        str_rows.append(grp)

    # ── save ───────────────────────────────────────────────────────────────
    if outbase is not None:
        out = Path(outbase)
        out.mkdir(parents=True, exist_ok=True)

        csv_path = out / "summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Results table → {csv_path}")

        tex_path = out / "summary.tex"
        _write_latex_table(df, tex_path)
        print(f"  LaTeX table   → {tex_path}")

    return df


def _write_latex_table(df: "pd.DataFrame", path: Path) -> None:
    """
    Write a LaTeX booktabs table from the results DataFrame.

    Best values (marked with ``*``) are rendered in \\textbf{}.
    """
    col_labels = {
        "dataset":            "Dataset",
        "method":             "Method",
        "mmd":                "MMD $\\downarrow$",
        "sliced_wasserstein": "SW $\\downarrow$",
        "nan_rate":           "NaN rate $\\downarrow$",
        "coverage_fraction":  "Cov.\\,frac.",
        "mean_nearest_dist":  "Mean dist.",
    }

    def _fmt(val) -> str:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return "--"
        s = str(val)
        is_best = s.endswith("*")
        try:
            num = float(s.rstrip("*"))
            formatted = f"{num:.5f}"
        except ValueError:
            formatted = s.rstrip("*")
        return f"\\textbf{{{formatted}}}" if is_best else formatted

    visible_cols = [c for c in df.columns if c in col_labels]
    header = " & ".join(col_labels[c] for c in visible_cols) + " \\\\"

    lines = [
        "\\begin{table}[ht]",
        "  \\centering",
        "  \\caption{Nonlinear SDE — teacher comparison}",
        "  \\label{tab:nl_teacher_compare}",
        f"  \\begin{{tabular}}{{{'l' * len(visible_cols)}}}",
        "    \\toprule",
        f"    {header}",
        "    \\midrule",
    ]

    prev_ds = None
    for _, row in df.iterrows():
        if prev_ds is not None and row["dataset"] != prev_ds:
            lines.append("    \\midrule")
        cells = " & ".join(_fmt(row[c]) for c in visible_cols)
        lines.append(f"    {cells} \\\\")  
        prev_ds = row["dataset"]

    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]

    path.write_text("\n".join(lines) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Residual correction sweep
# ──────────────────────────────────────────────────────────────────────────────

def run_residual_sweep(
    dataset: str = "swissroll",
    outbase: str = "results/mirafzali_residual_sweep",
    alphas: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
    nw_bandwidth_scales: Sequence[float] = (1.0, 2.0, 4.0),
    knn_ks: Sequence[int] = (128, 256, 512, 1024),
    knn_bandwidth_scales: Sequence[float] = (0.5, 1.0, 2.0),
    cfg: NonlinearSDEConfig = DEFAULT_NL_CFG,
    times: List[float] = TIMES,
    seed: int = 0,
    device: Optional[str] = None,
    mirafzali_mode: bool = True,
    n_bins: int = 60,
    n_paths: int = 50_000,
    n_epochs: int = 8_000,
    batch_size: int = 4_096,
    lr: float = 2e-4,
    n_samples_reverse: int = 20_000,
    n_steps_per_unit: int = _N_STEPS_PER_UNIT,
    gamma_reg: float = 1e-3,
    hidden: int = 512,
    n_blocks: int = 6,
    num_frequencies: int = 16,
    weight_decay: float = 1e-5,
) -> dict:
    """
    Sweep residual-correction hyper-parameters on a single dataset.

    Strategy (efficient):
    1. Simulate once (shared sim_cache).
    2. Train the base mirafzali model once (shared across all residual configs).
    3. Compute residuals once (shared r_tr).
    4. For each (method, alpha, bandwidth) config: build ResidualCorrectionModel
       and run reverse sampling — no re-simulation, no re-training.

    Configs swept:
    - ``mirafzali``          : one run (no alpha)
    - ``mirafzali_residual_binned``  : alphas
    - ``mirafzali_residual_nw``      : alphas × nw_bandwidth_scales
    - ``mirafzali_residual_knn_nw``  : alphas × knn_ks × knn_bandwidth_scales

    Output layout
    -------------
    ``{outbase}/{dataset}/{method}/{config_key}/``
        metrics.json, reverse_samples.png, residual_field.png

    ``{outbase}/{dataset}/``
        sweep_summary.json, best_by_mmd.json, best_by_sw.json

    Returns
    -------
    results : nested dict  ``{config_key: metrics}``
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if mirafzali_mode:
        n_paths   = 25_000
        n_epochs  = 2_500
        batch_size = 2_048
        lr        = 3e-4

    torch.manual_seed(seed)
    np.random.seed(seed)

    outdir_ds = Path(outbase) / dataset
    outdir_ds.mkdir(parents=True, exist_ok=True)

    # ── 1. Forward simulation (shared) ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  run_residual_sweep  dataset={dataset}  device={device}")
    print(f"{'='*60}", flush=True)
    print("Simulating forward paths (Malliavin) …", flush=True)
    sim_cache = simulate_all_times_nl(
        times, dataset, cfg,
        n_paths=n_paths, n_steps_per_unit=n_steps_per_unit,
        gamma_reg=gamma_reg, device=device,
        correction="approx",
    )

    # ── 2. Build training data + train base model (shared) ─────────────────
    print("Building teacher dataset (mirafzali) …", flush=True)
    dataset_tuple = build_training_dataset_nl(
        sim_cache, "mirafzali", teacher_eval_points="raw_points",
    )
    if dataset_tuple is None:
        raise RuntimeError("No valid teacher points found for mirafzali.")
    t_tr, x_tr, H_tr, _ = dataset_tuple
    print(f"  training points: {t_tr.shape[0]:,}", flush=True)

    print("Training MirafzaliSkorokhodNet …", flush=True)
    base_model = train_mirafzali_skorokhod_net(
        t_tr, x_tr, H_tr,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        hidden=hidden,
        n_blocks=n_blocks,
        num_frequencies=num_frequencies,
        device=device,
    )

    # Save base model
    base_outdir = outdir_ds / "mirafzali" / "default"
    base_outdir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": base_model.state_dict(), "dataset": dataset, "method": "mirafzali"},
        base_outdir / "model.pt",
    )

    # ── 3. Compute residuals (shared) ──────────────────────────────────────
    print("Computing residuals …", flush=True)
    r_tr, resid_diags = compute_residuals_nl(base_model, t_tr, x_tr, H_tr, device=device)
    print(
        f"  var_H={resid_diags['var_H']:.4f}  "
        f"var_r={resid_diags['var_residual']:.4f}  "
        f"mean‖r‖={resid_diags['mean_residual_norm']:.4f}",
        flush=True,
    )

    # Group by discrete time step (used by ResidualCorrectionModel)
    unique_times = sorted({float(t.item()) for t in t_tr.unique()})
    X_T_by_t, R_by_t = [], []
    for Tc in unique_times:
        mask = (t_tr.cpu() - Tc).abs() < 1e-5
        X_T_by_t.append(x_tr.cpu()[mask])
        R_by_t.append(r_tr[mask])

    # ── 4. Helper: run reverse sampling for a corrected model ──────────────
    sampler     = get_sampler(dataset)
    n_steps_rev = 250

    centers_np = None
    if dataset == "8gmm":
        _, centers_t = sample_8gmm(1, device="cpu")
        centers_np   = centers_t.numpy()

    def _run_one(config_key: str, corrected_model, extra_meta: dict) -> dict:
        """Reverse-sample, compute metrics, save outputs for one config."""
        cdir = outdir_ds / config_key
        cdir.mkdir(parents=True, exist_ok=True)

        corrected_model.eval()
        corrected_model = corrected_model.to(device)

        # Residual field plot
        _plot_residual_field(corrected_model, times[-1], device, cdir / "residual_field.png")

        # Reverse sampling
        torch.manual_seed(seed)
        X_T_start = sample_stationary_nl(n_samples_reverse, cfg, dim=2, device=device, clamp=20.0)
        samples   = reverse_euler_nl(corrected_model, X_T_start, cfg, n_steps=n_steps_rev)

        nan_mask = torch.isnan(samples).any(dim=1)
        nan_rate = float(nan_mask.float().mean().item())
        clean    = samples[~nan_mask].cpu().numpy()
        print(f"    {config_key}: {len(clean):,} clean  nan={nan_rate:.3f}", flush=True)

        # Scatter
        result_r = sampler(5_000, device="cpu")
        ref_np   = (result_r[0] if isinstance(result_r, tuple) else result_r).numpy()
        _plot_scatter(
            clean[:5_000],
            f"Residual sweep — {dataset}/{config_key}",
            cdir / "reverse_samples.png",
            ref_np=ref_np,
        )

        m = compute_metrics_nl(
            clean, nan_rate,
            dataset_name=dataset,
            centers_np=centers_np,
            rng=np.random.default_rng(seed),
        )
        m.update(extra_meta)
        m.update(resid_diags)
        m["config_key"] = config_key
        with open(cdir / "metrics.json", "w") as f:
            json.dump(m, f, indent=2)
        return m

    all_results: dict = {}

    # ── 5a. Baseline mirafzali (no residual) ──────────────────────────────
    print("\n--- mirafzali baseline ---", flush=True)
    torch.manual_seed(seed)
    X_T_start = sample_stationary_nl(n_samples_reverse, cfg, dim=2, device=device, clamp=20.0)
    samples_b = reverse_euler_nl(
        base_model.eval().to(device), X_T_start, cfg, n_steps=n_steps_rev,
    )
    nan_mask = torch.isnan(samples_b).any(dim=1)
    nan_rate = float(nan_mask.float().mean().item())
    clean_b  = samples_b[~nan_mask].cpu().numpy()
    cdir_b   = outdir_ds / "mirafzali" / "default"
    m_b = compute_metrics_nl(
        clean_b, nan_rate, dataset_name=dataset,
        centers_np=centers_np, rng=np.random.default_rng(seed),
    )
    m_b.update({"method": "mirafzali", "config_key": "mirafzali/default"})
    with open(cdir_b / "metrics.json", "w") as f:
        json.dump(m_b, f, indent=2)
    all_results["mirafzali/default"] = m_b
    result_r = sampler(5_000, device="cpu")
    ref_np   = (result_r[0] if isinstance(result_r, tuple) else result_r).numpy()
    _plot_scatter(clean_b[:5_000], f"mirafzali baseline — {dataset}", cdir_b / "reverse_samples.png", ref_np=ref_np)
    print(f"    mirafzali/default: mmd={m_b['mmd_rbf']:.5f}  sw={m_b['sliced_wasserstein']:.5f}", flush=True)

    # ── 5b. mirafzali_residual_binned (sweep alpha) ───────────────────────
    print("\n--- mirafzali_residual_binned ---", flush=True)
    for alpha in alphas:
        key = f"mirafzali_residual_binned/a{alpha:.2f}"
        cm  = ResidualCorrectionModel(
            base_model, unique_times, X_T_by_t, R_by_t,
            mode="binned", alpha=alpha, n_bins=n_bins,
        )
        m = _run_one(key, cm, {"method": "mirafzali_residual_binned", "residual_alpha": alpha})
        all_results[key] = m
        print(f"    {key}: mmd={m['mmd_rbf']:.5f}  sw={m['sliced_wasserstein']:.5f}", flush=True)

    # ── 5c. mirafzali_residual_nw (sweep alpha × nw_bandwidth_scale) ──────
    print("\n--- mirafzali_residual_nw ---", flush=True)
    for alpha in alphas:
        for bw in nw_bandwidth_scales:
            key = f"mirafzali_residual_nw/a{alpha:.2f}_bw{bw:.1f}"
            cm  = ResidualCorrectionModel(
                base_model, unique_times, X_T_by_t, R_by_t,
                mode="nw", alpha=alpha, nw_bandwidth_scale=bw,
            )
            m = _run_one(key, cm, {
                "method": "mirafzali_residual_nw",
                "residual_alpha": alpha, "nw_bandwidth_scale": bw,
            })
            all_results[key] = m
            print(f"    {key}: mmd={m['mmd_rbf']:.5f}  sw={m['sliced_wasserstein']:.5f}", flush=True)

    # ── 5d. mirafzali_residual_knn_nw (sweep alpha × knn_k × knn_bw) ─────
    print("\n--- mirafzali_residual_knn_nw ---", flush=True)
    for alpha in alphas:
        for k in knn_ks:
            for bw in knn_bandwidth_scales:
                key = f"mirafzali_residual_knn_nw/a{alpha:.2f}_k{k}_bw{bw:.1f}"
                cm  = ResidualCorrectionModel(
                    base_model, unique_times, X_T_by_t, R_by_t,
                    mode="knn_nw", alpha=alpha,
                    knn_k=k, knn_bandwidth_scale=bw,
                )
                m = _run_one(key, cm, {
                    "method": "mirafzali_residual_knn_nw",
                    "residual_alpha": alpha,
                    "residual_knn_k": k,
                    "residual_knn_bandwidth_scale": bw,
                })
                all_results[key] = m
                print(f"    {key}: mmd={m['mmd_rbf']:.5f}  sw={m['sliced_wasserstein']:.5f}", flush=True)

    # ── 6. Aggregate outputs ───────────────────────────────────────────────
    sweep_path = outdir_ds / "sweep_summary.json"
    with open(sweep_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Sweep summary → {sweep_path}", flush=True)

    finite_results = {
        k: v for k, v in all_results.items()
        if isinstance(v.get("mmd_rbf"), float) and math.isfinite(v["mmd_rbf"])
    }

    def _best(metric_key: str) -> dict:
        if not finite_results:
            return {}
        best_key = min(finite_results, key=lambda k: finite_results[k][metric_key])
        return {"best_config": best_key, "best_value": finite_results[best_key][metric_key],
                "all_values": {k: v[metric_key] for k, v in finite_results.items()}}

    best_mmd = _best("mmd_rbf")
    best_sw  = _best("sliced_wasserstein")

    for fname, data in [("best_by_mmd.json", best_mmd), ("best_by_sw.json", best_sw)]:
        p = outdir_ds / fname
        with open(p, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  {fname} → {p}", flush=True)

    if best_mmd:
        print(f"\n  Best MMD : {best_mmd['best_config']}  = {best_mmd['best_value']:.5f}")
    if best_sw:
        print(f"  Best SW  : {best_sw['best_config']}  = {best_sw['best_value']:.5f}")

    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# Batch runner
# ──────────────────────────────────────────────────────────────────────────────

def run_phase_b(
    datasets: Sequence[str] = ("8gmm",),
    methods: Sequence[str] = ("Mirafzali", "raw", "binned", "nw", "knn_nw"),
    cfg: NonlinearSDEConfig = DEFAULT_NL_CFG,
    outbase: str = "results/mirafzali_nonlinear_equal_points",
    teacher_eval_points: str = "raw_points",
    **kwargs,
) -> dict:
    """
    Run Phase B experiments for all (dataset, method) combinations.

    Returns nested dict: results[dataset][method] = metrics.
    """
    results: dict = {}

    for ds in datasets:
        results.setdefault(ds, {})
        for method in methods:
            m = run_experiment_nl(
                ds, method, cfg=cfg, outbase=outbase,
                teacher_eval_points=teacher_eval_points,
                **kwargs,
            )
            results[ds][method] = m

    summary_path = Path(outbase) / "phase_b_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nPhase B summary → {summary_path}")

    build_results_table(results, outbase=outbase)
    return results


def run_mirafzali_baseline(
    datasets: Sequence[str] = ("swissroll", "8gmm"),
    cfg: NonlinearSDEConfig = DEFAULT_NL_CFG,
    outbase: str = "results/mirafzali_nonlinear_baseline",
    **kwargs,
) -> dict:
    """
    Run the faithful Mirafzali Algorithm 6/7 baseline for all datasets.

    Uses ``method='mirafzali'`` (all n_paths, equal weights, plain MSE)
    and writes results under ``outbase/{dataset}/mirafzali/``.

    Returns nested dict: results[dataset]['mirafzali'] = metrics.
    """
    return run_phase_b(
        datasets=datasets,
        methods=("mirafzali",),
        cfg=cfg,
        outbase=outbase,
        teacher_eval_points="raw_points",
        **kwargs,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Multi-seed evaluation runner
# ──────────────────────────────────────────────────────────────────────────────

# Each config is a dict of keyword arguments forwarded to run_experiment_nl.
# ``_key`` is used as the directory name; ``method`` must be present.
_MULTISEED_CONFIGS = [
    {
        "_key": "baseline",
        "method": "mirafzali",
    },
    {
        "_key": "residual_nw__a1.0_bw1.0",
        "method": "mirafzali_residual_nw",
        "residual_alpha": 1.0,
        "nw_bandwidth_scale": 1.0,
    },
    {
        "_key": "residual_knn_nw__a1.0_k128_bw0.5",
        "method": "mirafzali_residual_knn_nw",
        "residual_alpha": 1.0,
        "residual_knn_k": 128,
        "residual_knn_bandwidth_scale": 0.5,
    },
    {
        "_key": "residual_knn_nw__a1.0_k1024_bw0.5",
        "method": "mirafzali_residual_knn_nw",
        "residual_alpha": 1.0,
        "residual_knn_k": 1024,
        "residual_knn_bandwidth_scale": 0.5,
    },
]


def run_residual_multiseed_eval(
    dataset: str = "swissroll",
    outbase: str = "results/mirafzali_residual_multiseed",
    seeds: Sequence[int] = tuple(range(10)),
    configs: Optional[Sequence[dict]] = None,
    cfg: NonlinearSDEConfig = DEFAULT_NL_CFG,
    times: List[float] = TIMES,
    correction: str = "approx",
    hidden: int = 2048,
    n_blocks: int = 6,
    num_frequencies: int = 16,
    n_paths: int = 25_000,
    n_epochs: int = 2_500,
    batch_size: int = 1_024,
    lr: float = 2e-4,
    n_samples_reverse: int = 20_000,
    n_steps_per_unit: int = _N_STEPS_PER_UNIT,
    gamma_reg: float = 1e-3,
    weight_decay: float = 1e-5,
    reverse_init: str = "stationary",
    device: Optional[str] = None,
    mirafzali_mode: bool = False,
) -> dict:
    """
    Run the same set of residual-correction configs across multiple seeds and
    collect statistics to assess whether improvements are stable.

    For each (seed, config) pair the full pipeline is run via
    ``run_experiment_nl``; all other hyper-parameters are shared.

    Output layout
    -------------
    ``{outbase}/{dataset}/seed{seed}/{config_key}/``
        metrics.json  (written by run_experiment_nl)

    ``{outbase}/{dataset}/``
        raw_results.csv
        raw_results.json
        summary.csv
        summary.json
        paired_tests.json

    Parameters
    ----------
    configs : list of dicts, each containing at least ``'method'`` and ``'_key'``.
              Defaults to ``_MULTISEED_CONFIGS``.

    Returns
    -------
    dict with keys ``'raw'``, ``'summary'``, ``'paired_tests'``.
    """
    import time as _time_module

    try:
        from scipy.stats import ttest_rel as _ttest_rel
        _has_scipy = True
    except ImportError:
        _has_scipy = False

    if configs is None:
        configs = _MULTISEED_CONFIGS
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    outdir_ds = Path(outbase) / dataset
    outdir_ds.mkdir(parents=True, exist_ok=True)

    # Shared kwargs forwarded to run_experiment_nl (can be overridden per config).
    _shared = dict(
        dataset=dataset,
        cfg=cfg,
        times=times,
        correction=correction,
        hidden=hidden,
        n_blocks=n_blocks,
        num_frequencies=num_frequencies,
        n_paths=n_paths,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        n_samples_reverse=n_samples_reverse,
        n_steps_per_unit=n_steps_per_unit,
        gamma_reg=gamma_reg,
        weight_decay=weight_decay,
        reverse_init=reverse_init,
        device=device,
        mirafzali_mode=mirafzali_mode,
    )

    raw_rows: list = []

    for seed in seeds:
        for cfg_entry in configs:
            cfg_entry = dict(cfg_entry)          # copy so we can pop
            config_key = cfg_entry.pop("_key")
            method      = cfg_entry["method"]

            run_outdir = outdir_ds / f"seed{seed}" / config_key
            run_outdir.mkdir(parents=True, exist_ok=True)

            print(f"\n{'─'*60}")
            print(f"  seed={seed}  config={config_key}")
            print(f"{'─'*60}", flush=True)

            t0 = _time_module.perf_counter()
            metrics = run_experiment_nl(
                **_shared,
                **cfg_entry,
                seed=seed,
                _outdir_override=str(run_outdir),
            )
            elapsed = _time_module.perf_counter() - t0

            row: dict = {
                "seed":       seed,
                "config_key": config_key,
                "method":     method,
                "mmd_rbf":    metrics.get("mmd_rbf"),
                "sliced_wasserstein": metrics.get("sliced_wasserstein"),
                "nan_rate":   metrics.get("nan_rate"),
                "elapsed_seconds": round(elapsed, 2),
            }
            for diag_key in ("var_H", "var_residual", "mean_residual_norm"):
                row[diag_key] = metrics.get(diag_key)
            raw_rows.append(row)

    # ── Save raw results ───────────────────────────────────────────────────
    raw_json_path = outdir_ds / "raw_results.json"
    with open(raw_json_path, "w") as f:
        json.dump(raw_rows, f, indent=2)

    raw_df = pd.DataFrame(raw_rows)
    raw_csv_path = outdir_ds / "raw_results.csv"
    raw_df.to_csv(raw_csv_path, index=False)
    print(f"\n  raw_results → {raw_json_path}", flush=True)

    # ── Summary per config ─────────────────────────────────────────────────
    summary_rows: list = []
    for cfg_entry in configs:
        cfg_entry = dict(cfg_entry)
        config_key = cfg_entry.pop("_key")
        method      = cfg_entry.get("method", "")
        sub = raw_df[raw_df["config_key"] == config_key]
        if sub.empty:
            continue
        mmds  = sub["mmd_rbf"].dropna().tolist()
        sws   = sub["sliced_wasserstein"].dropna().tolist()
        nrs   = sub["nan_rate"].dropna().tolist()
        elaps = sub["elapsed_seconds"].dropna().tolist()
        summary_rows.append({
            "config_key":    config_key,
            "method":        method,
            "n_seeds":       len(sub),
            "mmd_mean":      float(np.mean(mmds))  if mmds  else None,
            "mmd_std":       float(np.std(mmds, ddof=1)) if len(mmds) > 1 else None,
            "sw_mean":       float(np.mean(sws))   if sws   else None,
            "sw_std":        float(np.std(sws, ddof=1))  if len(sws) > 1 else None,
            "nan_rate_mean": float(np.mean(nrs))   if nrs   else None,
            "mean_runtime":  float(np.mean(elaps)) if elaps else None,
        })

    summary_json_path = outdir_ds / "summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary_rows, f, indent=2)
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = outdir_ds / "summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"  summary       → {summary_json_path}", flush=True)

    # ── Paired tests vs baseline ───────────────────────────────────────────
    baseline_key = configs[0] if isinstance(configs[0], str) else \
        {k: v for k, v in (list(configs[0].items()) if hasattr(configs[0], "items") else [])}
    # Find baseline config_key (first entry)
    _first = dict(configs[0])
    baseline_cfg_key = _first.pop("_key")
    baseline_sub = raw_df[raw_df["config_key"] == baseline_cfg_key].sort_values("seed")

    paired_tests: dict = {}
    for cfg_entry in configs[1:]:
        cfg_entry  = dict(cfg_entry)
        config_key = cfg_entry.pop("_key")
        sub = raw_df[raw_df["config_key"] == config_key].sort_values("seed")

        # Align on seeds present in both
        common_seeds = sorted(
            set(baseline_sub["seed"].tolist()) & set(sub["seed"].tolist())
        )
        if not common_seeds:
            paired_tests[config_key] = {"error": "no_common_seeds"}
            continue

        base_mmds = baseline_sub[baseline_sub["seed"].isin(common_seeds)].sort_values("seed")["mmd_rbf"].tolist()
        cand_mmds = sub[sub["seed"].isin(common_seeds)].sort_values("seed")["mmd_rbf"].tolist()
        base_sws  = baseline_sub[baseline_sub["seed"].isin(common_seeds)].sort_values("seed")["sliced_wasserstein"].tolist()
        cand_sws  = sub[sub["seed"].isin(common_seeds)].sort_values("seed")["sliced_wasserstein"].tolist()

        def _rel_improvement(base_vals, cand_vals):
            """Mean relative improvement: (base - cand) / base; positive = better."""
            diffs = []
            for b, c in zip(base_vals, cand_vals):
                if b is not None and c is not None and b != 0:
                    diffs.append((b - c) / abs(b))
            return float(np.mean(diffs)) if diffs else None

        mmd_rel  = _rel_improvement(base_mmds, cand_mmds)
        sw_rel   = _rel_improvement(base_sws,  cand_sws)

        mmd_pval: Optional[float] = None
        sw_pval:  Optional[float] = None
        if _has_scipy and len(common_seeds) >= 2:
            try:
                _b = [x for x in base_mmds if x is not None]
                _c = [x for x in cand_mmds if x is not None]
                if len(_b) == len(_c) and len(_b) >= 2:
                    mmd_pval = float(_ttest_rel(_b, _c).pvalue)
            except Exception:
                pass
            try:
                _b = [x for x in base_sws if x is not None]
                _c = [x for x in cand_sws if x is not None]
                if len(_b) == len(_c) and len(_b) >= 2:
                    sw_pval = float(_ttest_rel(_b, _c).pvalue)
            except Exception:
                pass

        paired_tests[config_key] = {
            "n_paired_seeds":          len(common_seeds),
            "mean_relative_mmd_improvement": mmd_rel,
            "mean_relative_sw_improvement":  sw_rel,
            "mmd_ttest_pvalue":        mmd_pval,
            "sw_ttest_pvalue":         sw_pval,
            "scipy_available":         _has_scipy,
        }

    paired_path = outdir_ds / "paired_tests.json"
    with open(paired_path, "w") as f:
        json.dump(paired_tests, f, indent=2)
    print(f"  paired_tests  → {paired_path}", flush=True)

    # ── Print summary table ────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"{'Config':<45}  {'MMD mean±std':>18}  {'SW mean±std':>18}")
    print(f"{'─'*60}")
    for row in summary_rows:
        mmd_str = (
            f"{row['mmd_mean']:.5f}±{row['mmd_std']:.5f}"
            if row["mmd_std"] is not None else
            f"{row['mmd_mean']:.5f}"
        ) if row["mmd_mean"] is not None else "N/A"
        sw_str = (
            f"{row['sw_mean']:.5f}±{row['sw_std']:.5f}"
            if row["sw_std"] is not None else
            f"{row['sw_mean']:.5f}"
        ) if row["sw_mean"] is not None else "N/A"
        print(f"{row['config_key']:<45}  {mmd_str:>18}  {sw_str:>18}")

    return {
        "raw":          raw_rows,
        "summary":      summary_rows,
        "paired_tests": paired_tests,
    }


if __name__ == "__main__":
    run_phase_b(
        datasets=("8gmm",),
        methods=("raw", "binned", "nw", "knn_nw"),
        n_paths=50_000,
        n_epochs=8_000,
        n_samples_reverse=20_000,
    )
