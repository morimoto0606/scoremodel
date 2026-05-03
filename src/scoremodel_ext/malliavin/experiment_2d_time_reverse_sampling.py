import math
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")

try:
    from .models import TimeScoreMLP2D
    from .sde_2d import sample_8gmm, drift, sigma_schedule
except ImportError:
    from models import TimeScoreMLP2D
    from sde_2d import sample_8gmm, drift, sigma_schedule


@torch.no_grad()
def sample_forward_terminal(
    n_samples=30_000,
    T=1.0,
    n_steps=300,
    sigma=0.45,
    sigma_min=0.15,
    sigma_max=1.20,
    device="cuda",
):
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)

    x, centers = sample_8gmm(n_samples, device=device)
    x0 = x.clone()

    for k in range(n_steps):
        t_mid = (k + 0.5) * dt

        sigma_k = sigma_schedule(
            t_mid,
            T,
            sigma=sigma,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )

        x = x + drift(x) * dt + sigma_k * sqrt_dt * torch.randn_like(x)

    return x, x0, centers


@torch.no_grad()
def reverse_sample(
    model,
    n_samples=30_000,
    T=1.0,
    n_steps=800,
    sigma=0.45,
    sigma_min=0.15,
    sigma_max=1.20,
    device="cuda",
):
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)

    x, x0_true, centers = sample_forward_terminal(
        n_samples=n_samples,
        T=T,
        n_steps=300,
        sigma=sigma,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        device=device,
    )

    x_terminal = x.detach().cpu().clone()

    traj = []
    save_steps = {0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1}

    for k in reversed(range(n_steps)):
        t_current = (k + 0.5) * dt
        t_tensor = torch.full((n_samples,), t_current, device=device)

        sigma_k = sigma_schedule(
            t_current,
            T,
            sigma=sigma,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )

        score = model(t_tensor, x)

        score = torch.nan_to_num(score, nan=0.0, posinf=30.0, neginf=-30.0)
        score = score.clamp(-30.0, 30.0)

        reverse_drift = -drift(x) + sigma_k**2 * score
        reverse_drift = torch.nan_to_num(
            reverse_drift,
            nan=0.0,
            posinf=80.0,
            neginf=-80.0,
        )
        reverse_drift = reverse_drift.clamp(-80.0, 80.0)

        x = x + reverse_drift * dt + sigma_k * sqrt_dt * torch.randn_like(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=6.0, neginf=-6.0)
        x = x.clamp(-6.0, 6.0)

        if k in save_steps:
            traj.append((k, x.detach().cpu().clone()))

    return x.detach().cpu(), x_terminal, x0_true.detach().cpu(), centers.detach().cpu(), traj


def plot_scatter(x, centers, title, path, lim=3.2):
    plt.figure(figsize=(7, 7))
    plt.scatter(x[:, 0], x[:, 1], s=2, alpha=0.22, label="samples")
    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        s=90,
        marker="x",
        label="initial modes",
    )
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.gca().set_aspect("equal")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"saved: {path}")


def mode_coverage_stats(x, centers):
    d2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(dim=2)
    assign = d2.argmin(dim=1)
    counts = torch.bincount(assign, minlength=8).float()
    proportions = counts / counts.sum()
    nearest_dist = torch.sqrt(d2.min(dim=1).values)

    return {
        "counts": counts.tolist(),
        "proportions": proportions.tolist(),
        "min_prop": proportions.min().item(),
        "max_prop": proportions.max().item(),
        "mean_nearest_dist": nearest_dist.mean().item(),
        "std_nearest_dist": nearest_dist.std().item(),
    }


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    sigma_min = 0.15
    sigma_max = 1.20

    ckpt = torch.load(
        "results/2d_time_malliavin_binned/time_score_mlp_2d.pt",
        map_location=device,
    )

    model = TimeScoreMLP2D().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    final_x, terminal_x, x0_true, centers, traj = reverse_sample(
        model,
        n_samples=30_000,
        T=1.0,
        n_steps=800,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        device=device,
    )

    outdir = Path("results/2d_time_malliavin_reverse")
    outdir.mkdir(parents=True, exist_ok=True)

    plot_scatter(
        x0_true,
        centers,
        "Original 8-GMM samples",
        outdir / "original_samples.png",
    )

    plot_scatter(
        terminal_x,
        centers,
        "Forward terminal samples at T=1",
        outdir / "forward_terminal_samples.png",
    )

    plot_scatter(
        final_x,
        centers,
        "Time-dependent reverse samples",
        outdir / "reverse_final_samples.png",
    )

    for k, xk in traj:
        plot_scatter(
            xk,
            centers,
            f"Reverse trajectory step={k}",
            outdir / f"traj_step_{k:04d}.png",
        )

    print("original stats:", mode_coverage_stats(x0_true, centers))
    print("terminal stats:", mode_coverage_stats(terminal_x, centers))
    print("reverse stats:", mode_coverage_stats(final_x, centers))


if __name__ == "__main__":
    run()