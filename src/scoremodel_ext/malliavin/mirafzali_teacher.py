from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch

try:
    from .datasets_2d import get_sampler
    from .sde_2d import bin_teacher_2d, knn_nw_teacher_2d, nw_teacher_2d
    from .sde_nonlinear import NonlinearSDEConfig, simulate_malliavin_nl
except ImportError:
    from datasets_2d import get_sampler
    from sde_2d import bin_teacher_2d, knn_nw_teacher_2d, nw_teacher_2d
    from sde_nonlinear import NonlinearSDEConfig, simulate_malliavin_nl


DEFAULT_NL_CFG = NonlinearSDEConfig(
    k=1.0, sigma=1.0, a=0.0,
    beta_min=1.0, beta_max=25.0, T=1.0,
)

_N_STEPS_PER_UNIT = 250


def _n_steps_for(T: float, n_steps_per_unit: int = _N_STEPS_PER_UNIT) -> int:
    return max(10, round(T * n_steps_per_unit))


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
    sum0 = torch.bincount(flat_all, weights=H[:, 0].contiguous(), minlength=n_cells)
    sum1 = torch.bincount(flat_all, weights=H[:, 1].contiguous(), minlength=n_cells)
    avg_h0 = sum0 / counts.clamp_min(1.0)
    avg_h1 = sum1 / counts.clamp_min(1.0)

    # Look up bin average for each query point
    ix_q = (torch.bucketize(query_x[:, 0].contiguous(), x_edges) - 1).clamp(0, n_bins - 1)
    iy_q = (torch.bucketize(query_x[:, 1].contiguous(), y_edges) - 1).clamp(0, n_bins - 1)
    flat_q = ix_q * n_bins + iy_q

    sc = torch.stack([avg_h0[flat_q], avg_h1[flat_q]], dim=1)
    cc = torch.ones(query_x.shape[0], device=device)
    return query_x, sc, cc


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
        n = X_T.shape[0]
        idx = torch.randperm(n, device=X_T.device)[: min(n_raw, n)]
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
        n = X_T.shape[0]
        idx = torch.randperm(n, device=X_T.device)[: min(n_raw, n)]
        pts = X_T[idx]
        sc = H[idx]
        cc = torch.ones(pts.shape[0], device=X_T.device)
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
        X0 = result[0] if isinstance(result, tuple) else result
        n_steps = _n_steps_for(T, n_steps_per_unit)

        print(f"  Simulating T={T:.2f}  n_steps={n_steps} …", flush=True)
        X_T, H = simulate_malliavin_nl(
            X0, T, cfg, n_steps=n_steps, gamma_reg=gamma_reg,
            correction=correction,
        )
        cache.append((T, X_T, H))

    return cache


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
