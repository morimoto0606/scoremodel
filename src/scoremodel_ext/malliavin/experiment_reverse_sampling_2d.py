import math
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")


def sample_8gmm(n, radius=2.0, std=0.08, device="cuda"):
    angles = torch.arange(8, device=device) * 2 * math.pi / 8
    centers = radius * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    idx = torch.randint(0, 8, (n,), device=device)
    x0 = centers[idx] + std * torch.randn(n, 2, device=device)
    return x0, centers


def drift(x):
    r2 = (x * x).sum(dim=1, keepdim=True)
    return -0.1 * x - 0.01 * r2 * x


class ScoreMLP2D(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def sample_forward_terminal(
    n_samples=20_000,
    T=0.35,
    n_steps=120,
    sigma=0.45,
    device="cuda",
):
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)

    x, centers = sample_8gmm(n_samples, device=device)
    x0 = x.clone()

    for _ in range(n_steps):
        x = x + drift(x) * dt + sigma * sqrt_dt * torch.randn_like(x)

    return x, x0, centers


@torch.no_grad()
def reverse_sample(
    model,
    n_samples=20_000,
    T=0.35,
    n_steps=500,
    sigma=0.45,
    device="cuda",
):
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)

    x, x0_true, centers = sample_forward_terminal(
        n_samples=n_samples,
        T=T,
        n_steps=120,
        sigma=sigma,
        device=device,
    )

    x_terminal = x.detach().cpu().clone()

    traj = []
    save_steps = {0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1}

    for k in reversed(range(n_steps)):
        score = model(x)

        score = torch.nan_to_num(score, nan=0.0, posinf=20.0, neginf=-20.0)
        score = score.clamp(-20.0, 20.0)

        reverse_drift = -drift(x) + sigma**2 * score
        reverse_drift = torch.nan_to_num(reverse_drift, nan=0.0, posinf=50.0, neginf=-50.0)
        reverse_drift = reverse_drift.clamp(-50.0, 50.0)

        x = x + reverse_drift * dt + sigma * sqrt_dt * torch.randn_like(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0)
        x = x.clamp(-5.0, 5.0)

        if k in save_steps:
            traj.append((k, x.detach().cpu().clone()))

    return x.detach().cpu(), x_terminal, x0_true.detach().cpu(), centers.detach().cpu(), traj


def plot_scatter(x, centers, title, path, lim=3.2):
    plt.figure(figsize=(7, 7))
    plt.scatter(x[:, 0], x[:, 1], s=2, alpha=0.25, label="samples")
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
    # nearest mode assignment
    d2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(dim=2)
    assign = d2.argmin(dim=1)
    counts = torch.bincount(assign, minlength=8).float()
    proportions = counts / counts.sum()

    min_prop = proportions.min().item()
    max_prop = proportions.max().item()

    nearest_dist = torch.sqrt(d2.min(dim=1).values)

    return {
        "counts": counts.tolist(),
        "proportions": proportions.tolist(),
        "min_prop": min_prop,
        "max_prop": max_prop,
        "mean_nearest_dist": nearest_dist.mean().item(),
        "std_nearest_dist": nearest_dist.std().item(),
    }


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    ckpt_path = "results/2d_malliavin_binned_teacher/score_mlp_2d_binned.pt"
    ckpt = torch.load(ckpt_path, map_location=device)

    model = ScoreMLP2D().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    final_x, terminal_x, x0_true, centers, traj = reverse_sample(
        model,
        n_samples=20_000,
        T=0.35,
        n_steps=500,
        sigma=0.45,
        device=device,
    )

    outdir = Path("results/2d_malliavin_reverse")
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
        "Forward terminal samples X_T",
        outdir / "forward_terminal_samples.png",
    )

    plot_scatter(
        final_x,
        centers,
        "Reverse samples using binned Malliavin score",
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