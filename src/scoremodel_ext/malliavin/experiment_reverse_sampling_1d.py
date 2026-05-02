import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scoremodel_ext.malliavin.models import TimeScoreMLP

plt.style.use("seaborn-v0_8")

def b(x):
    return -x - x**3


@torch.no_grad()
def sample_forward_terminal(
    n_samples,
    T=1.0,
    n_steps=400,
    sigma=0.8,
    x0=1.5,
    device="cuda",
):
    dt = T / n_steps
    sqrt_dt = torch.sqrt(torch.tensor(dt, device=device))

    x = torch.full((n_samples,), x0, device=device)

    for _ in range(n_steps):
        x = x + b(x) * dt + sigma * sqrt_dt * torch.randn_like(x)

    return x


@torch.no_grad()
def reverse_sample(
    model,
    n_samples=5000,
    T=1.0,
    n_steps=400,
    sigma=0.8,
    x0=1.5,
    device="cuda",
):
    dt = T / n_steps
    sqrt_dt = torch.sqrt(torch.tensor(dt, device=device))

    # Forward simulation で本物の X_T を作って、そこから reverse を開始
    x = sample_forward_terminal(
        n_samples=n_samples,
        T=T,
        n_steps=n_steps,
        sigma=sigma,
        x0=x0,
        device=device,
    )

    x_terminal = x.detach().cpu().clone()

    traj = []
    save_steps = {0, 100, 200, 300, 399}

    for k in reversed(range(n_steps)):
        t = torch.full((n_samples,), (k + 1) * dt, device=device)
        score = model(t, x).squeeze(-1)
        score = torch.nan_to_num(score, nan=0.0, posinf=20.0, neginf=-20.0)
        score = score.clamp(-20.0, 20.0)

        reverse_drift = -b(x) + sigma**2 * score
        reverse_drift = torch.nan_to_num(reverse_drift, nan=0.0, posinf=50.0, neginf=-50.0)
        reverse_drift = reverse_drift.clamp(-50.0, 50.0)

        x = x + reverse_drift * dt + sigma * sqrt_dt * torch.randn_like(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0)
        x = x.clamp(-5.0, 5.0)

        if k in save_steps:
            traj.append((k, x.detach().cpu().clone()))

    return x.detach().cpu(), x_terminal, traj


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    ckpt = torch.load(
        "results/malliavin_time_corrected_1d/time_score_mlp.pt",
        map_location=device,
    )
    
    hidden = ckpt.get("hidden", 128)
    model = TimeScoreMLP(hidden=hidden).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    final_x, terminal_x, traj = reverse_sample(
        model,
        n_steps=1000,
        device=device,
    ) 
    outdir = Path("results/malliavin_reverse_1d")
    outdir.mkdir(parents=True, exist_ok=True)

    # forward terminal histogram
    plt.figure(figsize=(8, 5))
    plt.hist(terminal_x.numpy(), bins=80, density=True, alpha=0.75)
    plt.xlabel("x")
    plt.ylabel("density")
    plt.title("Forward terminal samples X_T")
    plt.tight_layout()
    plt.savefig(outdir / "forward_terminal_hist.png", dpi=220)
    plt.close()

    # final reverse histogram
    plt.figure(figsize=(8, 5))
    plt.hist(final_x.numpy(), bins=80, density=True, alpha=0.75)
    plt.axvline(1.5, linestyle="--", linewidth=2, label="true x0=1.5")
    plt.xlabel("x")
    plt.ylabel("density")
    plt.title("Reverse samples at t=0")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "reverse_final_hist.png", dpi=220)
    plt.close()

    print(f"saved: {outdir / 'forward_terminal_hist.png'}")
    print(f"saved: {outdir / 'reverse_final_hist.png'}")

    # trajectory histograms
    for k, xk in traj:
        plt.figure(figsize=(8, 5))
        plt.hist(xk.numpy(), bins=80, density=True, alpha=0.75)
        plt.xlabel("x")
        plt.ylabel("density")
        plt.title(f"Reverse trajectory step={k}")
        plt.tight_layout()
        plt.savefig(outdir / f"traj_step_{k:03d}.png", dpi=220)
        plt.close()

    print("saved trajectory histograms")

    print(
        f"terminal mean={terminal_x.mean().item():.4f}, "
        f"terminal std={terminal_x.std().item():.4f}"
    )
    print(
        f"final mean={final_x.mean().item():.4f}, "
        f"final std={final_x.std().item():.4f}"
    )


if __name__ == "__main__":
    run()