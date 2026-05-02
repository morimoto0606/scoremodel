import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use("seaborn-v0_8")


def b_np(x):
    return -x - x**3


def solve_fp_pde(
    T=1.0,
    sigma=0.8,
    x0=1.5,
    x_min=-3.0,
    x_max=3.0,
    nx=1200,
    nt=60000,
):
    """
    Solve Fokker-Planck:
      ∂_t p = -∂_x(b p) + 0.5 σ^2 ∂_{xx}p
    with narrow Gaussian initial approximation to delta_x0.
    """
    x = np.linspace(x_min, x_max, nx)
    dx = x[1] - x[0]
    dt = T / nt

    # delta initial condition approximated by narrow Gaussian
    eps0 = 0.03
    p = np.exp(-0.5 * ((x - x0) / eps0) ** 2)
    p /= np.trapezoid(p, x)

    D = 0.5 * sigma**2

    for _ in range(nt):
        bp = b_np(x) * p

        # central differences
        d_bp = np.zeros_like(p)
        d2_p = np.zeros_like(p)

        d_bp[1:-1] = (bp[2:] - bp[:-2]) / (2 * dx)
        d2_p[1:-1] = (p[2:] - 2 * p[1:-1] + p[:-2]) / dx**2

        p = p + dt * (-d_bp + D * d2_p)

        # crude boundary stabilization
        p[0] = 0.0
        p[-1] = 0.0
        p = np.maximum(p, 1e-300)
        p /= np.trapezoid(p, x)

    dp = np.zeros_like(p)
    dp[1:-1] = (p[2:] - p[:-2]) / (2 * dx)

    score = dp / np.maximum(p, 1e-300)
    return x, p, score


def b(x):
    return -x - x**3


def bp(x):
    return -1.0 - 3.0 * x**2


def bpp(x):
    return -6.0 * x


def simulate_malliavin(
    n_paths=300_000,
    n_steps=400,
    T=1.0,
    sigma=0.8,
    x0=1.5,
    n_bins=160,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    dt = T / n_steps
    sqrt_dt = torch.sqrt(torch.tensor(dt, device=device))

    X = torch.full((n_paths,), x0, device=device)
    Y = torch.ones(n_paths, device=device)
    Z = torch.zeros(n_paths, device=device)

    dB_all = torch.empty((n_paths, n_steps), device=device)
    Y_all = torch.empty((n_paths, n_steps), device=device)
    Z_all = torch.empty((n_paths, n_steps), device=device)

    for k in range(n_steps):
        Y_all[:, k] = Y
        Z_all[:, k] = Z

        dB = sqrt_dt * torch.randn(n_paths, device=device)
        dB_all[:, k] = dB

        X_old = X
        Y_old = Y
        Z_old = Z

        X = X_old + b(X_old) * dt + sigma * dB
        Y = Y_old + bp(X_old) * Y_old * dt
        Z = Z_old + (bpp(X_old) * Y_old**2 + bp(X_old) * Z_old) * dt

    X_T = X
    Y_T = Y
    Z_T = Z

    eps = 1e-8
    Y_safe = torch.where(Y_all.abs() < eps, eps * torch.sign(Y_all + eps), Y_all)

    invY2 = 1.0 / (Y_safe**2)
    Z_over_Y3 = Z_all / (Y_safe**3)

    B0_tail = torch.flip(torch.cumsum(torch.flip(invY2, dims=[1]), dim=1), dims=[1]) * dt
    B1_tail = torch.flip(torch.cumsum(torch.flip(Z_over_Y3, dims=[1]), dim=1), dims=[1]) * dt

    A = sigma**2 * B0_tail[:, 0]
    gamma = (Y_T**2) * A

    u = (sigma * Y_T[:, None] / Y_safe) / gamma[:, None]
    ito_part = torch.sum(u * dB_all, dim=1)

    DtY_T = sigma * (Z_T[:, None] / Y_safe - Y_T[:, None] * Z_all / (Y_safe**2))

    DtA = -2.0 * sigma**3 * (
        B1_tail / Y_safe - (Z_all / (Y_safe**2)) * B0_tail
    )

    Dtu = u * (
        -DtY_T / torch.where(Y_T[:, None].abs() < eps, eps, Y_T[:, None])
        -DtA / torch.where(A[:, None].abs() < eps, eps, A[:, None])
    )

    correction = torch.sum(Dtu, dim=1) * dt
    delta_u_corrected = ito_part - correction

    x_min, x_max = torch.quantile(X_T, torch.tensor([0.005, 0.995], device=device))
    bins = torch.linspace(x_min, x_max, n_bins + 1, device=device)
    centers = 0.5 * (bins[:-1] + bins[1:])

    idx = torch.bucketize(X_T, bins) - 1
    mask = (idx >= 0) & (idx < n_bins)
    idx = idx[mask]

    counts = torch.bincount(idx, minlength=n_bins).float()
    sums_ito = torch.bincount(idx, weights=ito_part[mask], minlength=n_bins)
    sums_corr = torch.bincount(idx, weights=delta_u_corrected[mask], minlength=n_bins)

    score_ito = -(sums_ito / counts.clamp_min(1.0))
    score_corr = -(sums_corr / counts.clamp_min(1.0))

    valid = counts > 100

    return {
        "centers": centers[valid].detach().cpu().numpy(),
        "score_ito": score_ito[valid].detach().cpu().numpy(),
        "score_corr": score_corr[valid].detach().cpu().numpy(),
        "counts": counts[valid].detach().cpu().numpy(),
        "stats": {
            "X_mean": X_T.mean().item(),
            "X_std": X_T.std().item(),
            "gamma_mean": gamma.mean().item(),
            "gamma_min": gamma.min().item(),
            "ito_mean": ito_part.mean().item(),
            "correction_mean": correction.mean().item(),
        },
    }


def run():
    T = 1.0
    sigma = 0.8
    x0 = 1.5

    mc = simulate_malliavin(T=T, sigma=sigma, x0=x0)

    print("solving PDE...")
    x_pde, p_pde, score_pde = solve_fp_pde(T=T, sigma=sigma, x0=x0)

    # interpolate PDE score at MC bin centers
    score_true_at_centers = np.interp(mc["centers"], x_pde, score_pde)
    print(np.nanmin(score_pde), np.nanmax(score_pde))
    print(np.nanmin(p_pde), np.nanmax(p_pde))
    p_at_centers = np.interp(mc["centers"], x_pde, p_pde)

    valid_pde = np.isfinite(score_true_at_centers) & (p_at_centers > 1e-4)

    rmse_ito = np.sqrt(
        np.mean((mc["score_ito"][valid_pde] - score_true_at_centers[valid_pde]) ** 2))
    rmse_corr = np.sqrt(
        np.mean((mc["score_corr"][valid_pde] - score_true_at_centers[valid_pde]) ** 2))

    print(f"RMSE Ito only     = {rmse_ito:.6f}")
    print(f"RMSE Corrected    = {rmse_corr:.6f}")
    print(mc["stats"])

    outdir = Path("results/malliavin_nonlinear_1d_pde_compare")
    outdir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(x_pde, score_pde, label="PDE true score", linewidth=2)
    plt.scatter(
        mc["centers"][valid_pde],
        mc["score_ito"][valid_pde],
        s=10,
        alpha=0.55,
        label="Ito part only",
    )

    plt.scatter(
        mc["centers"][valid_pde],
        mc["score_corr"][valid_pde],
        s=10,
        alpha=0.55,
        label="Skorokhod corrected",
    )

    plt.ylim(-10, 8)
    plt.xlabel("y")
    plt.ylabel("score")
    plt.title("Nonlinear additive SDE: Malliavin score vs PDE score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "score_vs_pde.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(x_pde, p_pde, linewidth=2)
    plt.xlabel("y")
    plt.ylabel("density")
    plt.title("Fokker-Planck density at T")
    plt.tight_layout()
    plt.savefig(outdir / "pde_density.png", dpi=220)
    plt.close()

    print(f"saved: {outdir / 'score_vs_pde.png'}")
    print(f"saved: {outdir / 'pde_density.png'}")


if __name__ == "__main__":
    run()