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


def vp_forward(x0, t, beta=2.0):
    alpha = torch.exp(-0.5 * beta * t)[:, None]
    sigma = torch.sqrt(1.0 - torch.exp(-beta * t))[:, None]
    eps = torch.randn_like(x0)
    xt = alpha * x0 + sigma * eps
    return xt, eps, sigma


class NoiseMLP(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, t, x):
        inp = torch.cat([t[:, None], x], dim=1)
        return self.net(inp)


def train(
    n_steps=30_000,
    batch_size=4096,
    lr=2e-4,
    beta=2.0,
    t_min=0.05,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    model = NoiseMLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    ema_loss = None

    for step in range(1, n_steps + 1):
        x0, _ = sample_8gmm(batch_size, device=device)

        t = t_min + (1.0 - t_min) * torch.rand(batch_size, device=device)

        xt, eps, _ = vp_forward(x0, t, beta=beta)

        pred_eps = model(t, xt)
        loss = ((pred_eps - eps) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        ema_loss = loss.item() if ema_loss is None else 0.98 * ema_loss + 0.02 * loss.item()

        if step % 1000 == 0:
            print(f"step={step:6d}, loss={loss.item():.6e}, ema={ema_loss:.6e}")

    return model


@torch.no_grad()
def plot_score_field(model, beta=2.0, device="cuda"):
    outdir = Path("results/2d_8gmm_baseline")
    outdir.mkdir(parents=True, exist_ok=True)

    grid_n = 35
    lim = 3.2
    xs = torch.linspace(-lim, lim, grid_n, device=device)
    ys = torch.linspace(-lim, lim, grid_n, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    points = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

    for t_value in [0.05, 0.2, 0.5, 1.0]:
        t = torch.full((points.shape[0],), t_value, device=device)

        pred_eps = model(t, points)
        sigma = torch.sqrt(1.0 - torch.exp(-beta * t))[:, None]
        score = -pred_eps / sigma

        pts = points.detach().cpu()
        sc = score.detach().cpu()

        x0, centers = sample_8gmm(5000, device=device)
        xt, _, _ = vp_forward(
            x0,
            torch.full((5000,), t_value, device=device),
            beta=beta,
        )

        plt.figure(figsize=(7, 7))
        plt.scatter(
            xt[:, 0].detach().cpu(),
            xt[:, 1].detach().cpu(),
            s=2,
            alpha=0.25,
            label=f"forward samples t={t_value}",
        )
        plt.quiver(
            pts[:, 0],
            pts[:, 1],
            sc[:, 0],
            sc[:, 1],
            angles="xy",
            scale_units="xy",
            scale=18,
            width=0.0025,
            alpha=0.8,
        )
        plt.scatter(
            centers[:, 0].detach().cpu(),
            centers[:, 1].detach().cpu(),
            s=80,
            marker="x",
            label="original modes",
        )
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
        plt.gca().set_aspect("equal")
        plt.title(f"Learned VP score field from noise prediction, t={t_value}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.tight_layout()
        path = outdir / f"score_field_t_{t_value:.2f}.png"
        plt.savefig(path, dpi=220)
        plt.close()
        print(f"saved: {path}")


@torch.no_grad()
def plot_denoising(model, beta=2.0, device="cuda"):
    outdir = Path("results/2d_8gmm_baseline")
    outdir.mkdir(parents=True, exist_ok=True)

    for t_value in [0.05, 0.2, 0.5, 1.0]:
        n = 6000
        x0, centers = sample_8gmm(n, device=device)
        t = torch.full((n,), t_value, device=device)
        xt, eps, sigma = vp_forward(x0, t, beta=beta)

        pred_eps = model(t, xt)
        alpha = torch.exp(-0.5 * beta * t)[:, None]

        x0_hat = (xt - sigma * pred_eps) / alpha

        plt.figure(figsize=(7, 7))
        plt.scatter(
            xt[:, 0].detach().cpu(),
            xt[:, 1].detach().cpu(),
            s=2,
            alpha=0.18,
            label="noisy xt",
        )
        plt.scatter(
            x0_hat[:, 0].detach().cpu(),
            x0_hat[:, 1].detach().cpu(),
            s=2,
            alpha=0.35,
            label="denoised x0_hat",
        )
        plt.scatter(
            centers[:, 0].detach().cpu(),
            centers[:, 1].detach().cpu(),
            s=80,
            marker="x",
            label="true modes",
        )
        plt.xlim(-3.2, 3.2)
        plt.ylim(-3.2, 3.2)
        plt.gca().set_aspect("equal")
        plt.title(f"Denoising check, t={t_value}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.tight_layout()
        path = outdir / f"denoise_t_{t_value:.2f}.png"
        plt.savefig(path, dpi=220)
        plt.close()
        print(f"saved: {path}")


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    beta = 2.0

    model = train(beta=beta, device=device)

    outdir = Path("results/2d_8gmm_baseline")
    outdir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), outdir / "noise_mlp.pt")
    print(f"saved: {outdir / 'noise_mlp.pt'}")

    plot_score_field(model, beta=beta, device=device)
    plot_denoising(model, beta=beta, device=device)


if __name__ == "__main__":
    run()