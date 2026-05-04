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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

try:
    from .datasets_2d import get_sampler, sample_8gmm
    from .models import TimeScoreMLP2D
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
    from models import TimeScoreMLP2D
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply a teacher to (X_T, H) pairs from the nonlinear SDE simulation.

    Parameters
    ----------
    method : "raw" | "binned" | "nw" | "knn_nw"
    X_T    : (n, 2) terminal forward samples
    H      : (n, 2) Malliavin score weights

    Returns
    -------
    pts    : (m, 2) query positions
    scores : (m, 2) estimated scores
    counts : (m,)   per-point weights
    """
    if method == "raw":
        n   = X_T.shape[0]
        idx = torch.randperm(n, device=X_T.device)[:min(n_raw, n)]
        pts = X_T[idx]
        sc  = H[idx]
        cc  = torch.ones(pts.shape[0], device=X_T.device)
        return pts, sc, cc

    if method == "binned":
        return bin_teacher_2d(X_T, H, n_bins=n_bins, min_count=min_count)

    # NW and kNN-NW use the binned grid as query positions for consistency
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

    raise ValueError(
        f"Unknown teacher method {method!r}. "
        f"Choose from 'raw', 'binned', 'nw', 'knn_nw'."
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
) -> dict:
    """
    Full pipeline for one (dataset, method) combination with the nonlinear SDE.

    Returns metrics dict.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    outdir = Path(outbase) / dataset / method
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  dataset={dataset}  method={method}  device={device}")
    print(f"{'='*60}", flush=True)

    # ── forward simulation ─────────────────────────────────────────────────
    print("Simulating forward paths (Malliavin) …", flush=True)
    sim_cache = simulate_all_times_nl(
        times, dataset, cfg,
        n_paths=n_paths, n_steps_per_unit=n_steps_per_unit,
        gamma_reg=gamma_reg, device=device,
    )

    # ── build training dataset ─────────────────────────────────────────────
    print("Building teacher dataset …", flush=True)
    dataset_tuple = build_training_dataset_nl(
        sim_cache, method,
        n_raw=n_raw, n_bins=n_bins, min_count=min_count, knn_k=knn_k,
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

    # ── train MLP ──────────────────────────────────────────────────────────
    print("Training score MLP …", flush=True)
    model = train_score_mlp(
        t_tr, x_tr, s_tr, c_tr,
        n_epochs=n_epochs, batch_size=batch_size, lr=lr, device=device,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "dataset": dataset,
            "method": method,
            "sde": "nonlinear",
            "reverse_init": reverse_init,
            "cfg": cfg.__dict__ if hasattr(cfg, "__dict__") else str(cfg),
        },
        outdir / "model.pt",
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
    n_steps_rev = _n_steps_for(cfg.T, n_steps_per_unit)

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
# Batch runner
# ──────────────────────────────────────────────────────────────────────────────

def run_phase_b(
    datasets: Sequence[str] = ("8gmm",),
    methods: Sequence[str] = ("raw", "binned", "nw", "knn_nw"),
    cfg: NonlinearSDEConfig = DEFAULT_NL_CFG,
    outbase: str = "results/mirafzali_nonlinear",
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
            m = run_experiment_nl(ds, method, cfg=cfg, outbase=outbase, **kwargs)
            results[ds][method] = m

    summary_path = Path(outbase) / "phase_b_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nPhase B summary → {summary_path}")
    return results


if __name__ == "__main__":
    run_phase_b(
        datasets=("8gmm",),
        methods=("raw", "binned", "nw", "knn_nw"),
        n_paths=50_000,
        n_epochs=8_000,
        n_samples_reverse=20_000,
    )
