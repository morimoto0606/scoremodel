import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")
from pathlib import Path


def b(x):
    return -x - x**3


def bp(x):
    return -1.0 - 3.0 * x**2


def run(
    n_paths=500_000,
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

    # first variation Y = dX/dx0
    Y = torch.ones(n_paths, device=device)

    # Malliavin covariance gamma_T = int_0^T (D_s X_T)^2 ds
    # For additive noise: D_s X_T = Y_T Y_s^{-1} sigma.
    # We store integral of (sigma / Y_s)^2 ds, then multiply by Y_T^2.
    gamma_core = torch.zeros(n_paths, device=device)

    dB_all = torch.empty((n_paths, n_steps), device=device)
    invY_sigma_all = torch.empty((n_paths, n_steps), device=device)

    for k in range(n_steps):
        invY_sigma = sigma / Y
        gamma_core += invY_sigma**2 * dt
        invY_sigma_all[:, k] = invY_sigma

        dB = sqrt_dt * torch.randn(n_paths, device=device)
        dB_all[:, k] = dB

        # Euler for X
        X = X + b(X) * dt + sigma * dB

        # variation: dY = b'(X)Y dt
        # use old/new ambiguity is O(dt); this is fine for first experiment
        Y = Y + bp(X) * Y * dt

    X_T = X
    Y_T = Y
    gamma = (Y_T**2) * gamma_core

    # covering field u_s = D_s X_T / gamma = Y_T * (sigma/Y_s) / gamma
    u = (Y_T[:, None] * invY_sigma_all) / gamma[:, None]

    # additive/adapted approximation: delta(u) ≈ int u_s dB_s
    # In nonlinear case u is anticipating because gamma,Y_T depend on future.
    # This is a first diagnostic estimator; later we add Skorokhod correction.
    delta_u_ito_part = torch.sum(u * dB_all, dim=1)

    # score(y) = - E[delta(u) | X_T = y]
    x_min, x_max = torch.quantile(X_T, torch.tensor([0.005, 0.995], device=device))
    bins = torch.linspace(x_min, x_max, n_bins + 1, device=device)
    idx = torch.bucketize(X_T, bins) - 1
    mask = (idx >= 0) & (idx < n_bins)

    idx = idx[mask]
    delta_u = delta_u_ito_part[mask]
    X_used = X_T[mask]

    counts = torch.bincount(idx, minlength=n_bins).float()
    sums = torch.bincount(idx, weights=delta_u, minlength=n_bins)

    score_hat = -(sums / counts.clamp_min(1.0))
    centers = 0.5 * (bins[:-1] + bins[1:])

    valid = counts > 100

    outdir = Path("results/malliavin_nonlinear_1d")
    outdir.mkdir(parents=True, exist_ok=True)

    centers_cpu = centers[valid].detach().cpu()
    score_hat_cpu = score_hat[valid].detach().cpu()
    counts_cpu = counts[valid].detach().cpu()

    plt.figure(figsize=(7, 5))
    plt.scatter(centers_cpu, score_hat_cpu, s=8, label="Malliavin score estimate")
    plt.axhline(0.0, linewidth=1)
    plt.legend()
    plt.title("Nonlinear additive SDE score estimate")
    plt.xlabel("y")
    plt.ylabel("estimated score")
    plt.tight_layout()
    plt.savefig(outdir / "nonlinear_score_estimate.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(centers_cpu, counts_cpu, s=8)
    plt.title("Bin counts")
    plt.xlabel("y")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(outdir / "bin_counts.png", dpi=200)
    plt.close()

    print(f"saved: {outdir / 'nonlinear_score_estimate.png'}")
    print(f"saved: {outdir / 'bin_counts.png'}")
    print(f"X_T mean={X_T.mean().item():.4f}, std={X_T.std().item():.4f}")
    print(f"gamma mean={gamma.mean().item():.4e}, min={gamma.min().item():.4e}")


if __name__ == "__main__":
    run()