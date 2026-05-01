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


def bpp(x):
    return -6.0 * x


def run(
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
    Y = torch.ones(n_paths, device=device)       # first variation dX/dx0
    Z = torch.zeros(n_paths, device=device)      # second variation d2X/dx0^2

    dB_all = torch.empty((n_paths, n_steps), device=device)
    Y_all = torch.empty((n_paths, n_steps), device=device)
    Z_all = torch.empty((n_paths, n_steps), device=device)

    for k in range(n_steps):
        # left point values
        Y_all[:, k] = Y
        Z_all[:, k] = Z

        dB = sqrt_dt * torch.randn(n_paths, device=device)
        dB_all[:, k] = dB

        X_old = X
        Y_old = Y
        Z_old = Z

        # Euler
        X = X_old + b(X_old) * dt + sigma * dB

        # additive noise: dY = b'(X)Y dt
        Y = Y_old + bp(X_old) * Y_old * dt

        # dZ = {b''(X)Y^2 + b'(X)Z}dt
        Z = Z_old + (bpp(X_old) * Y_old**2 + bp(X_old) * Z_old) * dt

    X_T = X
    Y_T = Y
    Z_T = Z

    eps = 1e-8
    Y_safe = Y_all.clamp_min(eps)

    # A = sigma^2 int_0^T Y_s^{-2} ds
    invY2 = 1.0 / (Y_safe**2)
    Z_over_Y3 = Z_all / (Y_safe**3)

    # reverse cumulative integrals:
    # B0_t = int_t^T Y_r^{-2} dr
    # B1_t = int_t^T Z_r Y_r^{-3} dr
    B0_tail = torch.flip(torch.cumsum(torch.flip(invY2, dims=[1]), dim=1), dims=[1]) * dt
    B1_tail = torch.flip(torch.cumsum(torch.flip(Z_over_Y3, dims=[1]), dim=1), dims=[1]) * dt

    A = sigma**2 * B0_tail[:, 0]
    gamma = (Y_T**2) * A

    # u_t = D_t X_T / gamma
    # D_t X_T = sigma Y_T / Y_t
    u = (sigma * Y_T[:, None] / Y_safe) / gamma[:, None]

    ito_part = torch.sum(u * dB_all, dim=1)

    # D_t Y_T = sigma * ( Z_T / Y_t - Y_T Z_t / Y_t^2 )
    DtY_T = sigma * (Z_T[:, None] / Y_safe - Y_T[:, None] * Z_all / (Y_safe**2))

    # D_t A = -2 sigma^3 [ (1/Y_t) int_t^T Z_r/Y_r^3 dr
    #                      - (Z_t/Y_t^2) int_t^T 1/Y_r^2 dr ]
    DtA = -2.0 * sigma**3 * (
        B1_tail / Y_safe - (Z_all / (Y_safe**2)) * B0_tail
    )

    # u_t = sigma / (Y_t Y_T A)
    # D_t u_t = u_t * ( - D_tY_T/Y_T - D_tA/A )
    Dtu = u * (
        -DtY_T / Y_T[:, None].clamp_min(eps)
        -DtA / A[:, None].clamp_min(eps)
    )

    correction = torch.sum(Dtu, dim=1) * dt

    # Skorokhod integral
    delta_u_corrected = ito_part - correction

    # scores:
    # score = - E[delta(u) | X_T = y]
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

    outdir = Path("results/malliavin_nonlinear_1d_corrected")
    outdir.mkdir(parents=True, exist_ok=True)

    centers_cpu = centers[valid].detach().cpu()
    score_ito_cpu = score_ito[valid].detach().cpu()
    score_corr_cpu = score_corr[valid].detach().cpu()
    counts_cpu = counts[valid].detach().cpu()

    plt.figure(figsize=(7, 5))
    plt.scatter(centers_cpu, score_ito_cpu, s=8, alpha=0.6, label="Ito part only")
    plt.scatter(centers_cpu, score_corr_cpu, s=8, alpha=0.6, label="Skorokhod corrected")
    plt.axhline(0.0, linewidth=1)
    plt.legend()
    plt.title("Nonlinear additive SDE score estimate")
    plt.xlabel("y")
    plt.ylabel("estimated score")
    plt.tight_layout()
    plt.savefig(outdir / "score_ito_vs_corrected.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7,5))
    sc = plt.scatter(
        centers_cpu,
        counts_cpu,
        c=counts_cpu,
        cmap="viridis",
        s=18,
        edgecolors="none"
    )
    plt.colorbar(sc, label="count")
    plt.title("Bin Counts")
    plt.xlabel("y")
    plt.ylabel("count")
    plt.tight_layout()
    plt.title("Bin counts")
    plt.xlabel("y")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(outdir / "bin_counts.png", dpi=200)
    plt.close()

    print(f"saved: {outdir / 'score_ito_vs_corrected.png'}")
    print(f"saved: {outdir / 'bin_counts.png'}")
    print(f"X_T mean={X_T.mean().item():.4f}, std={X_T.std().item():.4f}")
    print(f"gamma mean={gamma.mean().item():.4e}, min={gamma.min().item():.4e}")
    print(f"ito_part mean={ito_part.mean().item():.4e}, std={ito_part.std().item():.4e}")
    print(f"correction mean={correction.mean().item():.4e}, std={correction.std().item():.4e}")


if __name__ == "__main__":
    run()