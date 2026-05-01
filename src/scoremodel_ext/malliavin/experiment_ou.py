import torch
import matplotlib.pyplot as plt
from pathlib import Path


def run(
    n_paths=300_000,
    n_steps=200,
    T=1.0,
    a=1.0,
    sigma=1.0,
    x0=2.0,
    n_bins=120,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    dt = T / n_steps
    t_grid = torch.linspace(0.0, T, n_steps + 1, device=device)

    # OU: dX = -a X dt + sigma dB
    # X_T = exp(-aT)x0 + int_0^T sigma exp(-a(T-s)) dB_s
    m_T = torch.exp(torch.tensor(-a * T, device=device)) * x0

    s_left = t_grid[:-1]
    D = sigma * torch.exp(-a * (T - s_left))  # D_s X_T
    gamma = torch.sum(D**2 * dt)              # Malliavin covariance

    dB = torch.sqrt(torch.tensor(dt, device=device)) * torch.randn(
        n_paths, n_steps, device=device
    )

    X_T = m_T + torch.sum(D[None, :] * dB, dim=1)

    # covering field u_s = D_s X_T / gamma
    # delta(u) = int u_s dB_s since deterministic/adapted
    delta_u = torch.sum((D / gamma)[None, :] * dB, dim=1)

    # Malliavin score: score(y) = - E[delta(u) | X_T = y]
    x_min, x_max = torch.quantile(X_T, torch.tensor([0.005, 0.995], device=device))
    bins = torch.linspace(x_min, x_max, n_bins + 1, device=device)
    idx = torch.bucketize(X_T, bins) - 1
    mask = (idx >= 0) & (idx < n_bins)

    idx = idx[mask]
    delta_u = delta_u[mask]
    X_T_used = X_T[mask]

    counts = torch.bincount(idx, minlength=n_bins).float()
    sums = torch.bincount(idx, weights=delta_u, minlength=n_bins)

    score_hat = -(sums / counts.clamp_min(1.0))
    centers = 0.5 * (bins[:-1] + bins[1:])

    # true transition score
    v_T = gamma
    score_true = -(centers - m_T) / v_T

    valid = counts > 50

    outdir = Path("results/malliavin_ou")
    outdir.mkdir(parents=True, exist_ok=True)

    centers_cpu = centers[valid].detach().cpu()
    score_hat_cpu = score_hat[valid].detach().cpu()
    score_true_cpu = score_true[valid].detach().cpu()

    rmse = torch.sqrt(torch.mean((score_hat[valid] - score_true[valid]) ** 2))
    print(f"RMSE = {rmse.item():.6f}")

    plt.figure(figsize=(7, 5))
    plt.plot(centers_cpu, score_true_cpu, label="true score")
    plt.scatter(centers_cpu, score_hat_cpu, s=8, label="Malliavin estimate")
    plt.legend()
    plt.title(f"OU score estimate, RMSE={rmse.item():.4g}")
    plt.xlabel("y")
    plt.ylabel("score")
    plt.tight_layout()
    plt.savefig(outdir / "ou_score_estimate.png", dpi=200)
    print(f"saved: {outdir / 'ou_score_estimate.png'}")


if __name__ == "__main__":
    run()