"""
Mirafzali reproduction — Phase A
=================================
Implements the full pipeline:
  dataset → linear SDE (forward) → Malliavin teacher → MLP training →
  reverse sampling → metrics → output

Datasets  : 8gmm, checkerboard, swissroll
SDE types : VE, VP, subVP
Methods   : raw, binned, knn_nw

Output layout
-------------
results/mirafzali_reproduction/{dataset}/{sde}/{method}/
  teacher_field.png
  score_field.png
  reverse_samples.png
  metrics.json

Run
---
python -m scoremodel_ext.malliavin.experiment_mirafzali
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

try:
    from .datasets_2d import get_sampler, sample_8gmm
    from .models import TimeScoreMLP2D
    from .sde_2d import bin_teacher_2d, knn_nw_teacher_2d
    from .sde_linear import (
        VEConfig, VPConfig, SubVPConfig,
        simulate_linear,
        reverse_sample_linear,
        ve_marginal_var, ve_sigma_at,
        vp_marginal_params,
    )
except ImportError:
    from datasets_2d import get_sampler, sample_8gmm
    from models import TimeScoreMLP2D
    from sde_2d import bin_teacher_2d, knn_nw_teacher_2d
    from sde_linear import (
        VEConfig, VPConfig, SubVPConfig,
        simulate_linear,
        reverse_sample_linear,
        ve_marginal_var, ve_sigma_at,
        vp_marginal_params,
    )

plt.style.use("seaborn-v0_8")


# ──────────────────────────────────────────────────────────────────────────────
# Default SDE configs
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_VE  = VEConfig(sigma_min=0.1, sigma_max=10.0, T=1.0)
DEFAULT_VP  = VPConfig(beta_min=0.1,  beta_max=20.0,  T=1.0)
DEFAULT_SUBVP = SubVPConfig(beta_min=0.1, beta_max=20.0, T=1.0)

_SDE_REGISTRY = {
    "ve":    DEFAULT_VE,
    "vp":    DEFAULT_VP,
    "subvp": DEFAULT_SUBVP,
}

TIMES: List[float] = [0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00]


# ──────────────────────────────────────────────────────────────────────────────
# Teacher methods
# ──────────────────────────────────────────────────────────────────────────────

def _apply_raw_teacher(X_T: torch.Tensor, H: torch.Tensor, n_raw: int = 20_000):
    """
    Raw teacher: subsample paths; use (X_T_i, H_i) directly as (point, score).

    Returns (pts, scores, counts).
    """
    n = X_T.shape[0]
    idx = torch.randperm(n, device=X_T.device)[:min(n_raw, n)]
    pts = X_T[idx]
    sc  = H[idx]
    cc  = torch.ones(pts.shape[0], device=X_T.device)
    return pts, sc, cc


def apply_teacher_linear(
    method: str,
    X_T: torch.Tensor,
    H: torch.Tensor,
    n_raw: int = 20_000,
    n_bins: int = 60,
    min_count: int = 20,
    knn_k: int = 256,
    bandwidth_scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply a teacher method to (X_T, H) pairs.

    Parameters
    ----------
    method        : "raw" | "binned" | "knn_nw"
    X_T           : (n, 2) terminal forward samples
    H             : (n, 2) Malliavin score weights

    Returns
    -------
    pts    : (m, 2) query positions
    scores : (m, 2) estimated scores
    counts : (m,)   per-point counts / weights
    """
    if method == "raw":
        return _apply_raw_teacher(X_T, H, n_raw=n_raw)

    if method == "binned":
        return bin_teacher_2d(X_T, H, n_bins=n_bins, min_count=min_count)

    if method == "knn_nw":
        # Use binned grid as query positions (consistent with existing code)
        pts, _, cc = bin_teacher_2d(X_T, H, n_bins=n_bins, min_count=min_count)
        if pts.shape[0] == 0:
            sc = pts.clone()
        else:
            sc = knn_nw_teacher_2d(
                X_T, H, pts,
                k=knn_k,
                bandwidth_scale=bandwidth_scale,
            )
        return pts, sc, cc

    raise ValueError(f"Unknown teacher method {method!r}. Choose from 'raw', 'binned', 'knn_nw'")


# ──────────────────────────────────────────────────────────────────────────────
# Simulation cache
# ──────────────────────────────────────────────────────────────────────────────

def simulate_all_times_linear(
    times: Sequence[float],
    dataset_name: str,
    sde_config,
    n_paths: int = 50_000,
    device: str = "cpu",
) -> List[Tuple[float, torch.Tensor, torch.Tensor]]:
    """
    For each T in *times*, sample X_0 from the dataset and simulate to X_T
    via the linear SDE.

    Returns
    -------
    List of (T, X_T, H) tuples, all tensors on *device*.
    """
    sampler = get_sampler(dataset_name)
    cache = []
    for T in times:
        result = sampler(n_paths, device=device)
        if isinstance(result, tuple):
            X0 = result[0]
        else:
            X0 = result
        X_T, H = simulate_linear(X0, T, sde_config)
        cache.append((T, X_T, H))
    return cache


# ──────────────────────────────────────────────────────────────────────────────
# Dataset builder
# ──────────────────────────────────────────────────────────────────────────────

def build_training_dataset(
    sim_cache: List[Tuple[float, torch.Tensor, torch.Tensor]],
    method: str,
    n_raw: int = 20_000,
    n_bins: int = 60,
    min_count: int = 20,
    knn_k: int = 256,
    bandwidth_scale: float = 1.0,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Apply teacher to each cached (T, X_T, H) and concatenate across time.

    Returns (t, x, s, c) or None if every time step yields zero valid points.
    """
    t_list, x_list, s_list, c_list = [], [], [], []
    device = sim_cache[0][1].device

    for T, X_T, H in sim_cache:
        pts, sc, cc = apply_teacher_linear(
            method, X_T, H,
            n_raw=n_raw, n_bins=n_bins, min_count=min_count,
            knn_k=knn_k, bandwidth_scale=bandwidth_scale,
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
# MLP training
# ──────────────────────────────────────────────────────────────────────────────

def train_score_mlp(
    t: torch.Tensor,
    x: torch.Tensor,
    s: torch.Tensor,
    c: torch.Tensor,
    n_epochs: int = 8_000,
    batch_size: int = 4_096,
    lr: float = 2e-4,
    device: str = "cpu",
    hidden: int = 256,
) -> TimeScoreMLP2D:
    """
    Train a time-conditional score MLP on (t, x) → s with weights c.

    Returns the best model (by weighted MSE on the full dataset).
    """
    model = TimeScoreMLP2D(hidden=hidden).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    w     = (c / c.mean()).to(device)
    n     = x.shape[0]

    best       = float("inf")
    best_state = None

    for ep in range(1, n_epochs + 1):
        idx  = torch.randint(0, n, (batch_size,), device=device)
        pred = model(t[idx], x[idx])
        loss = (w[idx, None] * (pred - s[idx]) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if ep % 500 == 0:
            with torch.no_grad():
                full_loss = (w[:, None] * (model(t, x) - s) ** 2).mean()
            if full_loss.item() < best:
                best       = full_loss.item()
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                print(f"  *** best updated: {best:.6e}", flush=True)
            print(f"  epoch={ep:5d}  loss={full_loss.item():.6e}  best={best:.6e}",
                  flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def _mmd_rbf(X: np.ndarray, Y: np.ndarray, n_sub: int = 2_000,
             sigma: float = 1.0, rng=None) -> float:
    if rng is None:
        rng = np.random.default_rng(42)
    ix = rng.choice(len(X), min(n_sub, len(X)), replace=False)
    iy = rng.choice(len(Y), min(n_sub, len(Y)), replace=False)
    Xs, Ys = X[ix], Y[iy]

    def gram(A, B):
        d2 = ((A[:, None] - B[None]) ** 2).sum(-1)
        return np.exp(-d2 / (2 * sigma ** 2))

    Kxx = gram(Xs, Xs)
    Kyy = gram(Ys, Ys)
    Kxy = gram(Xs, Ys)
    n, m = len(Xs), len(Ys)
    return float(
        (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
        + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
        - 2 * Kxy.mean()
    )


def _sliced_wasserstein(X: np.ndarray, Y: np.ndarray,
                        n_proj: int = 200, rng=None) -> float:
    if rng is None:
        rng = np.random.default_rng(42)
    d = X.shape[1]
    dirs = rng.standard_normal((n_proj, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    dists = []
    for v in dirs:
        px = np.sort(X @ v)
        py = np.sort(Y @ v)
        n = min(len(px), len(py))
        dists.append(np.abs(px[:n] - py[:n]).mean())
    return float(np.mean(dists))


def _mode_coverage_8gmm(
    samples_np: np.ndarray,
    centers_np: np.ndarray,
    threshold: float = 0.01,
) -> dict:
    d2 = ((samples_np[:, None, :] - centers_np[None, :, :]) ** 2).sum(-1)
    assign   = d2.argmin(axis=1)
    n_modes  = centers_np.shape[0]
    counts   = np.bincount(assign, minlength=n_modes).astype(float)
    props    = counts / counts.sum()
    covered  = (props > threshold).sum()
    nearest  = np.sqrt(d2.min(axis=1))
    return {
        "n_covered":         int(covered),
        "n_modes":           int(n_modes),
        "coverage_fraction": float(covered) / n_modes,
        "min_proportion":    float(props.min()),
        "max_proportion":    float(props.max()),
        "mean_nearest_dist": float(nearest.mean()),
    }


def compute_metrics(
    samples_np: np.ndarray,
    nan_rate: float,
    dataset_name: str,
    centers_np: Optional[np.ndarray] = None,
    n_ref: int = 10_000,
    rng=None,
) -> dict:
    """
    Compute evaluation metrics.

    Parameters
    ----------
    samples_np   : (n, 2) generated samples (NaN-filtered)
    nan_rate     : fraction of NaN samples before filtering
    dataset_name : used to decide which reference to generate
    centers_np   : (k, 2) mode centres (required for 8gmm mode coverage)
    n_ref        : number of reference samples to draw
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Generate reference samples on CPU
    device = "cpu"
    sampler = get_sampler(dataset_name)
    result  = sampler(n_ref, device=device)
    ref_np  = (result[0] if isinstance(result, tuple) else result).numpy()

    metrics: dict = {
        "nan_rate":          float(nan_rate),
        "n_samples":         int(len(samples_np)),
        "mmd_rbf":           _mmd_rbf(samples_np, ref_np, rng=rng),
        "sliced_wasserstein": _sliced_wasserstein(samples_np, ref_np, rng=rng),
    }

    if dataset_name == "8gmm" and centers_np is not None:
        metrics["mode_coverage"] = _mode_coverage_8gmm(samples_np, centers_np)

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────

def _plot_scatter(x_np: np.ndarray, title: str, path: Path, lim: float = 3.5,
                  ref_np: Optional[np.ndarray] = None):
    fig, ax = plt.subplots(figsize=(6, 6))
    if ref_np is not None:
        ax.scatter(ref_np[:, 0], ref_np[:, 1], s=1, alpha=0.15,
                   color="tab:grey", label="reference")
    ax.scatter(x_np[:, 0], x_np[:, 1], s=2, alpha=0.3,
               color="tab:blue", label="generated")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(markerscale=4)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_teacher_field(
    pts_np: np.ndarray, sc_np: np.ndarray,
    T: float, title: str, path: Path,
    lim: float = 3.5,
):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pts_np[:, 0], pts_np[:, 1], s=1, alpha=0.4, c="tab:grey")
    step = max(1, len(pts_np) // 800)
    ax.quiver(
        pts_np[::step, 0], pts_np[::step, 1],
        sc_np[::step, 0],  sc_np[::step, 1],
        scale=None, scale_units="xy", alpha=0.6,
    )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Per-run pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_experiment(
    dataset: str,
    sde_type: str,
    method: str,
    times: List[float] = TIMES,
    n_paths: int = 50_000,
    n_epochs: int = 8_000,
    batch_size: int = 4_096,
    lr: float = 2e-4,
    n_bins: int = 60,
    min_count: int = 20,
    knn_k: int = 256,
    n_raw: int = 20_000,
    n_samples_reverse: int = 20_000,
    n_steps_reverse: int = 250,
    sde_config=None,
    outbase: str = "results/mirafzali_reproduction",
    device: Optional[str] = None,
    seed: int = 0,
) -> dict:
    """
    Full pipeline for one (dataset, sde_type, method) combination.

    Returns metrics dict.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if sde_config is None:
        sde_config = _SDE_REGISTRY.get(sde_type)
        if sde_config is None:
            raise ValueError(f"Unknown SDE type {sde_type!r}. "
                             f"Choose from {list(_SDE_REGISTRY)}")

    outdir = Path(outbase) / dataset / sde_type / method
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  dataset={dataset}  sde={sde_type}  method={method}")
    print(f"  device={device}  outdir={outdir}")
    print(f"{'='*60}", flush=True)

    # ── forward simulation ─────────────────────────────────────────────────
    print("Simulating forward paths …", flush=True)
    sim_cache = simulate_all_times_linear(
        times, dataset, sde_config, n_paths=n_paths, device=device,
    )

    # ── build training dataset ─────────────────────────────────────────────
    print("Building teacher dataset …", flush=True)
    dataset_tuple = build_training_dataset(
        sim_cache, method,
        n_raw=n_raw, n_bins=n_bins, min_count=min_count,
        knn_k=knn_k,
    )
    if dataset_tuple is None:
        print("  ERROR: no valid teacher points found, aborting.", flush=True)
        return {"error": "no_valid_teacher_points"}

    t_tr, x_tr, s_tr, c_tr = dataset_tuple
    print(f"  training points: {t_tr.shape[0]:,}", flush=True)

    # Plot teacher field at T_max
    T_last, X_T_last, H_last = sim_cache[-1]
    pts_np = x_tr[t_tr == T_last].cpu().numpy()
    sc_np  = s_tr[t_tr == T_last].cpu().numpy()
    if len(pts_np) > 0:
        _plot_teacher_field(
            pts_np, sc_np, T_last,
            f"Teacher field — {method}, T={T_last:.2f}",
            outdir / "teacher_field.png",
        )

    # ── train MLP ──────────────────────────────────────────────────────────
    print("Training MLP …", flush=True)
    model = train_score_mlp(
        t_tr, x_tr, s_tr, c_tr,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
    )
    torch.save(
        {"model_state_dict": model.state_dict(),
         "dataset": dataset, "sde_type": sde_type, "method": method},
        outdir / "model.pt",
    )

    # ── reverse sampling ───────────────────────────────────────────────────
    print("Reverse sampling …", flush=True)
    model.eval()
    model = model.to(device)
    samples = reverse_sample_linear(
        model, n_samples_reverse, sde_config,
        n_steps=n_steps_reverse, device=device,
    )

    nan_mask  = torch.isnan(samples).any(dim=1)
    nan_rate  = float(nan_mask.float().mean().item())
    clean     = samples[~nan_mask].numpy()
    print(f"  reverse: {len(clean):,} clean samples  (nan_rate={nan_rate:.3f})",
          flush=True)

    # ── metrics ────────────────────────────────────────────────────────────
    centers_np = None
    if dataset == "8gmm":
        _, centers_t = sample_8gmm(1, device="cpu")
        centers_np = centers_t.numpy()

    metrics = compute_metrics(
        clean, nan_rate,
        dataset_name=dataset,
        centers_np=centers_np,
        rng=np.random.default_rng(seed),
    )
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  mmd={metrics['mmd_rbf']:.5f}  "
          f"sw={metrics['sliced_wasserstein']:.5f}", flush=True)

    # ── sample scatter plot ────────────────────────────────────────────────
    sampler   = get_sampler(dataset)
    result_r  = sampler(5_000, device="cpu")
    ref_np    = (result_r[0] if isinstance(result_r, tuple) else result_r).numpy()
    _plot_scatter(
        clean[:5_000], f"Reverse samples — {dataset}/{sde_type}/{method}",
        outdir / "reverse_samples.png",
        ref_np=ref_np,
    )

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Batch runner
# ──────────────────────────────────────────────────────────────────────────────

def run_phase_a(
    datasets: Sequence[str] = ("8gmm",),
    sde_types: Sequence[str] = ("ve",),
    methods: Sequence[str] = ("raw", "binned", "knn_nw"),
    outbase: str = "results/mirafzali_reproduction",
    **kwargs,
) -> dict:
    """
    Run Phase A experiments for all (dataset, sde_type, method) combinations.

    Returns nested dict: results[dataset][sde][method] = metrics.
    """
    results: dict = {}
    for ds in datasets:
        results.setdefault(ds, {})
        for sde in sde_types:
            results[ds].setdefault(sde, {})
            for method in methods:
                m = run_experiment(ds, sde, method, outbase=outbase, **kwargs)
                results[ds][sde][method] = m

    summary_path = Path(outbase) / "phase_a_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nPhase A summary → {summary_path}")
    return results


if __name__ == "__main__":
    run_phase_a(
        datasets=("8gmm",),
        sde_types=("ve",),
        methods=("raw", "binned", "knn_nw"),
        n_paths=50_000,
        n_epochs=8_000,
        n_samples_reverse=20_000,
        n_steps_reverse=250,
    )
