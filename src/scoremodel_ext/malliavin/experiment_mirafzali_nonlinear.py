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
        sim_cache,
        method,
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
    if method == "mirafzali":
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

_METHOD_ORDER = ["raw", "binned", "nw", "knn_nw", "mirafzali"]


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


if __name__ == "__main__":
    run_phase_b(
        datasets=("8gmm",),
        methods=("raw", "binned", "nw", "knn_nw"),
        n_paths=50_000,
        n_epochs=8_000,
        n_samples_reverse=20_000,
    )
