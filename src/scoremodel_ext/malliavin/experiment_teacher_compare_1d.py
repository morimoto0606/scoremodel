"""
Spec 01: 1D NW Teacher Validation
==================================
Compare four score-estimation teachers for the 1D nonlinear SDE

    dX_t = (-X - X^3) dt + 0.8 dW_t,  X_0 = 1.5

at times [0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00].

Teachers:
  binned  – histogram bin average (reference)
  NW      – Nadaraya-Watson Gaussian kernel regression
  kNN-NW  – NW with kNN-adaptive bandwidth

Ground truth: Fokker-Planck PDE numerical solution.
"""

import math
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")

try:
    from .experiment_nonlinear_pde1d import solve_fp_pde
except ImportError:
    from experiment_nonlinear_pde1d import solve_fp_pde


# ── SDE coefficients ──────────────────────────────────────────────────────────

def _b(x):
    return -x - x ** 3


def _bp(x):
    return -1.0 - 3.0 * x ** 2


def _bpp(x):
    return -6.0 * x


# ── Raw Malliavin simulation (returns per-path values) ────────────────────────

@torch.no_grad()
def simulate_malliavin_raw(
    T=1.0,
    n_paths=300_000,
    n_steps=400,
    sigma=0.8,
    x0=1.5,
    device="cuda",
):
    """
    Forward SDE with first/second variation and Skorokhod correction.

    Returns
    -------
    X_T : (n_paths,)  terminal positions
    H   : (n_paths,)  Malliavin weight;  E[H | X_T = y] = score(y)
    """
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)

    X = torch.full((n_paths,), x0, dtype=torch.float32, device=device)
    Y = torch.ones(n_paths, dtype=torch.float32, device=device)
    Z = torch.zeros(n_paths, dtype=torch.float32, device=device)

    dB_all = torch.empty(n_paths, n_steps, dtype=torch.float32, device=device)
    Y_all  = torch.empty(n_paths, n_steps, dtype=torch.float32, device=device)
    Z_all  = torch.empty(n_paths, n_steps, dtype=torch.float32, device=device)

    for k in range(n_steps):
        Y_all[:, k] = Y
        Z_all[:, k] = Z

        dB = sqrt_dt * torch.randn(n_paths, dtype=torch.float32, device=device)
        dB_all[:, k] = dB

        X_old, Y_old, Z_old = X, Y, Z
        X = X_old + _b(X_old) * dt + sigma * dB
        Y = Y_old + _bp(X_old) * Y_old * dt
        Z = Z_old + (_bpp(X_old) * Y_old ** 2 + _bp(X_old) * Z_old) * dt

    X_T, Y_T, Z_T = X, Y, Z

    eps = 1e-8
    Y_safe = torch.where(Y_all.abs() < eps,
                         eps * Y_all.sign().clamp(min=1.0),
                         Y_all)

    invY2     = 1.0 / Y_safe ** 2
    Z_over_Y3 = Z_all / Y_safe ** 3

    B0_tail = torch.flip(torch.cumsum(torch.flip(invY2, [1]), 1), [1]) * dt
    B1_tail = torch.flip(torch.cumsum(torch.flip(Z_over_Y3, [1]), 1), [1]) * dt

    A     = sigma ** 2 * B0_tail[:, 0]
    gamma = Y_T ** 2 * A

    u        = (sigma * Y_T[:, None] / Y_safe) / gamma[:, None].clamp_min(eps)
    ito_part = (u * dB_all).sum(1)

    DtY_T = sigma * (Z_T[:, None] / Y_safe - Y_T[:, None] * Z_all / Y_safe ** 2)
    DtA   = -2.0 * sigma ** 3 * (
        B1_tail / Y_safe - (Z_all / Y_safe ** 2) * B0_tail
    )

    Y_T_s = torch.where(Y_T[:, None].abs() < eps, eps, Y_T[:, None])
    A_s   = torch.where(A[:, None].abs() < eps, eps, A[:, None])
    Dtu        = u * (-DtY_T / Y_T_s - DtA / A_s)
    correction = Dtu.sum(1) * dt

    H = -(ito_part - correction)
    return X_T, H


# ── Teacher estimators ────────────────────────────────────────────────────────

def binned_teacher_1d(X_T, H, n_bins=160, min_count=30):
    """Histogram bin average of H conditioned on X_T."""
    device = X_T.device
    x_min, x_max = torch.quantile(X_T, torch.tensor([0.005, 0.995], device=device))
    edges   = torch.linspace(x_min, x_max, n_bins + 1, device=device)
    centers = 0.5 * (edges[:-1] + edges[1:])

    idx  = torch.bucketize(X_T, edges) - 1
    mask = (idx >= 0) & (idx < n_bins)
    idx  = idx[mask]

    counts = torch.bincount(idx, minlength=n_bins).float()
    sums   = torch.bincount(idx, weights=H[mask], minlength=n_bins)
    score  = sums / counts.clamp_min(1.0)

    valid = counts >= min_count
    return centers[valid], score[valid], counts[valid]


def nw_teacher_1d(X_T, H, query_x, bandwidth=None, batch_size=256):
    """
    Nadaraya-Watson Gaussian kernel regression.
    bandwidth: Silverman's rule of thumb when None.
    """
    device = X_T.device
    n = X_T.shape[0]

    if bandwidth is None:
        iqr = torch.quantile(X_T, 0.75) - torch.quantile(X_T, 0.25)
        h = float(0.9 * torch.minimum(X_T.std(), iqr / 1.34) * (n ** -0.2))
    else:
        h = float(bandwidth)

    nq    = query_x.shape[0]
    score = torch.zeros(nq, device=device)

    for i in range(0, nq, batch_size):
        xq   = query_x[i : i + batch_size]
        diff = (xq[:, None] - X_T[None, :]) / h
        kw   = torch.exp(-0.5 * diff ** 2)
        score[i : i + batch_size] = (kw * H[None, :]).sum(1) / kw.sum(1).clamp_min(1e-12)

    return score


def knn_nw_teacher_1d(X_T, H, query_x, k=500, batch_size=256):
    """
    NW with per-query bandwidth = distance to k-th nearest neighbour.
    """
    device = X_T.device
    nq    = query_x.shape[0]
    score = torch.zeros(nq, device=device)

    for i in range(0, nq, batch_size):
        xq   = query_x[i : i + batch_size]
        diff = xq[:, None] - X_T[None, :]
        dabs = diff.abs()

        h = dabs.kthvalue(k, dim=1).values.clamp_min(1e-8)
        kw = torch.exp(-0.5 * (diff / h[:, None]) ** 2)
        score[i : i + batch_size] = (kw * H[None, :]).sum(1) / kw.sum(1).clamp_min(1e-12)

    return score


# ── Per-T comparison ──────────────────────────────────────────────────────────

def _rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def compare_teachers_at_T(
    T,
    n_paths=300_000,
    n_steps=400,
    sigma=0.8,
    x0=1.5,
    n_bins=160,
    knn_k=500,
    device="cuda",
):
    """
    Simulate, compute all three teachers, evaluate RMSE vs PDE.

    Returns a dict with RMSE values and arrays for plotting.
    """
    print(f"\n── T={T:.2f} ──────────────────────────────────────────────")

    # --- simulate ---
    t0 = time.perf_counter()
    X_T, H = simulate_malliavin_raw(
        T=T, n_paths=n_paths, n_steps=n_steps, sigma=sigma, x0=x0, device=device
    )
    print(f"  sim:    {time.perf_counter() - t0:.2f}s  "
          f"X_T ∈ [{X_T.min().item():.3f}, {X_T.max().item():.3f}]  "
          f"H_std={H.std().item():.3f}")

    # --- PDE ground truth ---
    t0 = time.perf_counter()
    x_pde, p_pde, score_pde = solve_fp_pde(T=T, sigma=sigma, x0=x0)
    print(f"  PDE:    {time.perf_counter() - t0:.2f}s")

    # --- binned teacher (shared query grid) ---
    t0 = time.perf_counter()
    centers, score_bin, counts = binned_teacher_1d(X_T, H, n_bins=n_bins)
    print(f"  binned: {time.perf_counter() - t0:.2f}s  "
          f"bins={centers.shape[0]}")

    query_x = centers  # same grid for NW and kNN-NW

    # --- NW teacher ---
    t0 = time.perf_counter()
    score_nw = nw_teacher_1d(X_T, H, query_x)
    print(f"  NW:     {time.perf_counter() - t0:.2f}s")

    # --- kNN-NW teacher ---
    t0 = time.perf_counter()
    score_knn = knn_nw_teacher_1d(X_T, H, query_x, k=knn_k)
    print(f"  kNN-NW: {time.perf_counter() - t0:.2f}s  k={knn_k}")

    # --- RMSE vs PDE ---
    cx       = centers.detach().cpu().numpy()
    p_interp = np.interp(cx, x_pde, p_pde)
    s_interp = np.interp(cx, x_pde, score_pde)
    valid    = np.isfinite(s_interp) & (p_interp > 1e-4)

    sb = score_bin.detach().cpu().numpy()
    sn = score_nw.detach().cpu().numpy()
    sk = score_knn.detach().cpu().numpy()

    r = {
        "T":             T,
        "query_x":       cx,
        "valid":         valid,
        "score_pde":     s_interp,
        "score_bin":     sb,
        "score_nw":      sn,
        "score_knn":     sk,
        "counts":        counts.detach().cpu().numpy(),
        "rmse_bin":      _rmse(sb[valid], s_interp[valid]),
        "rmse_nw":       _rmse(sn[valid], s_interp[valid]),
        "rmse_knn":      _rmse(sk[valid], s_interp[valid]),
        "x_pde":         x_pde,
        "score_pde_full": score_pde,
        "p_pde_full":    p_pde,
    }

    print(f"  RMSE → binned={r['rmse_bin']:.4f}  "
          f"NW={r['rmse_nw']:.4f}  kNN-NW={r['rmse_knn']:.4f}")

    return r


# ── Plotting ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def plot_comparison(r, outdir):
    T      = r["T"]
    cx     = r["query_x"]
    valid  = r["valid"]
    x_pde  = r["x_pde"]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(x_pde, r["score_pde_full"], "k-", linewidth=2, label="PDE (true)", zorder=5)
    ax.scatter(cx[valid], r["score_bin"][valid],  s=10, alpha=0.7, label=f"binned (RMSE={r['rmse_bin']:.4f})")
    ax.scatter(cx[valid], r["score_nw"][valid],   s=10, alpha=0.7, label=f"NW (RMSE={r['rmse_nw']:.4f})")
    ax.scatter(cx[valid], r["score_knn"][valid],  s=10, alpha=0.7, label=f"kNN-NW (RMSE={r['rmse_knn']:.4f})")

    ax.set_xlim(-3, 3)
    ax.set_ylim(-12, 8)
    ax.set_xlabel("y")
    ax.set_ylabel("score")
    ax.set_title(f"1D teacher comparison  T={T:.2f}")
    ax.legend(fontsize=8)
    plt.tight_layout()

    path = outdir / f"compare_T{T:.2f}.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  saved: {path}")


def plot_rmse_summary(all_results, outdir):
    times    = [r["T"] for r in all_results]
    rmse_bin = [r["rmse_bin"] for r in all_results]
    rmse_nw  = [r["rmse_nw"]  for r in all_results]
    rmse_knn = [r["rmse_knn"] for r in all_results]

    plt.figure(figsize=(8, 5))
    plt.plot(times, rmse_bin, "o-", label="binned")
    plt.plot(times, rmse_nw,  "s-", label="NW")
    plt.plot(times, rmse_knn, "^-", label="kNN-NW")
    plt.xlabel("T")
    plt.ylabel("RMSE vs PDE score")
    plt.title("Score estimation RMSE by teacher method")
    plt.legend()
    plt.tight_layout()

    path = outdir / "rmse_summary.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

TIMES = (0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00)


def run(
    times=TIMES,
    n_paths=300_000,
    n_steps=400,
    sigma=0.8,
    x0=1.5,
    n_bins=160,
    knn_k=500,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    outdir = Path("results/teacher_compare_1d")
    outdir.mkdir(parents=True, exist_ok=True)

    all_results = []
    total_start = time.perf_counter()

    for T in times:
        r = compare_teachers_at_T(
            T=T,
            n_paths=n_paths,
            n_steps=n_steps,
            sigma=sigma,
            x0=x0,
            n_bins=n_bins,
            knn_k=knn_k,
            device=device,
        )
        all_results.append(r)
        plot_comparison(r, outdir)

    plot_rmse_summary(all_results, outdir)

    # print RMSE table
    print("\n═══ RMSE summary ═══")
    print(f"{'T':>6}  {'binned':>10}  {'NW':>10}  {'kNN-NW':>10}")
    for r in all_results:
        print(f"{r['T']:>6.2f}  {r['rmse_bin']:>10.4f}  {r['rmse_nw']:>10.4f}  {r['rmse_knn']:>10.4f}")

    torch.save(
        [
            {k: v for k, v in r.items()
             if not isinstance(v, np.ndarray) or k in ("query_x", "score_bin", "score_nw", "score_knn", "score_pde")}
            for r in all_results
        ],
        outdir / "results.pt",
    )

    print(f"\ntotal elapsed: {time.perf_counter() - total_start:.1f}s")
    print(f"outputs: {outdir}/")


if __name__ == "__main__":
    run()
