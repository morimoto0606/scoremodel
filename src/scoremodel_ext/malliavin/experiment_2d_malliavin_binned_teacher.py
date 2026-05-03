import time
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")

try:
    from .models import ScoreMLP2D
    from .sde_2d import simulate_2d_malliavin_ito, bin_teacher_2d
except ImportError:
    from models import ScoreMLP2D
    from sde_2d import simulate_2d_malliavin_ito, bin_teacher_2d


def smooth_grid_scores(points, scores, counts):
    # まずは何もしない。2D smoothing は次段階で追加。
    return points, scores, counts


def train_mlp(
    points,
    scores,
    counts,
    n_epochs=8000,
    batch_size=2048,
    lr=2e-4,
    device="cuda",
):
    model = ScoreMLP2D().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    weights = counts / counts.mean()
    n = points.shape[0]

    best_loss = float("inf")
    best_state = None

    print(f"training on {n} binned teacher points...")

    start = time.perf_counter()

    for epoch in range(1, n_epochs + 1):
        idx = torch.randint(0, n, (batch_size,), device=device)

        pred = model(points[idx])
        loss = (weights[idx, None] * (pred - scores[idx]) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if epoch % 500 == 0:
            with torch.no_grad():
                full_pred = model(points)
                full_loss = (weights[:, None] * (full_pred - scores) ** 2).mean()

            if full_loss.item() < best_loss:
                best_loss = full_loss.item()
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                print(f"*** best checkpoint updated: {best_loss:.6e}")

            elapsed = time.perf_counter() - start
            print(
                f"epoch={epoch:5d}, "
                f"loss={full_loss.item():.6e}, "
                f"best={best_loss:.6e}, "
                f"elapsed={elapsed:.1f}s"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


@torch.no_grad()
def plot_results(model, X_T, centers0, points, scores, counts, device="cuda"):
    outdir = Path("results/2d_malliavin_binned_teacher")
    outdir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 7))
    plt.scatter(
        X_T[:, 0].detach().cpu(),
        X_T[:, 1].detach().cpu(),
        s=1,
        alpha=0.15,
        label="X_T samples",
    )
    plt.scatter(
        centers0[:, 0].detach().cpu(),
        centers0[:, 1].detach().cpu(),
        s=90,
        marker="x",
        label="initial modes",
    )
    plt.xlim(-3.2, 3.2)
    plt.ylim(-3.2, 3.2)
    plt.gca().set_aspect("equal")
    plt.title("2D nonlinear SDE terminal samples")
    plt.legend()
    plt.tight_layout()
    path = outdir / "terminal_samples.png"
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"saved: {path}")

    # binned teacher vector field
    plt.figure(figsize=(7, 7))
    sc = plt.scatter(
        points[:, 0].detach().cpu(),
        points[:, 1].detach().cpu(),
        c=counts.detach().cpu(),
        cmap="viridis",
        s=12,
        alpha=0.75,
    )
    plt.colorbar(sc, label="bin count")
    plt.quiver(
        points[:, 0].detach().cpu(),
        points[:, 1].detach().cpu(),
        scores[:, 0].detach().cpu(),
        scores[:, 1].detach().cpu(),
        angles="xy",
        scale_units="xy",
        scale=22,
        width=0.002,
        alpha=0.85,
    )
    plt.xlim(-3.2, 3.2)
    plt.ylim(-3.2, 3.2)
    plt.gca().set_aspect("equal")
    plt.title("Binned Malliavin teacher score field")
    plt.tight_layout()
    path = outdir / "binned_teacher_field.png"
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"saved: {path}")

    # learned score field on grid
    grid_n = 40
    lim = 3.2
    xs = torch.linspace(-lim, lim, grid_n, device=device)
    ys = torch.linspace(-lim, lim, grid_n, device=device)
    Xg, Yg = torch.meshgrid(xs, ys, indexing="xy")
    grid = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1)

    pred = model(grid)

    plt.figure(figsize=(7, 7))
    plt.scatter(
        X_T[:, 0].detach().cpu(),
        X_T[:, 1].detach().cpu(),
        s=1,
        alpha=0.10,
        label="X_T samples",
    )
    plt.quiver(
        grid[:, 0].detach().cpu(),
        grid[:, 1].detach().cpu(),
        pred[:, 0].detach().cpu(),
        pred[:, 1].detach().cpu(),
        angles="xy",
        scale_units="xy",
        scale=22,
        width=0.002,
        alpha=0.85,
    )
    plt.scatter(
        centers0[:, 0].detach().cpu(),
        centers0[:, 1].detach().cpu(),
        s=90,
        marker="x",
        label="initial modes",
    )
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.gca().set_aspect("equal")
    plt.title("MLP learned score field from binned Malliavin teacher")
    plt.legend()
    plt.tight_layout()
    path = outdir / "learned_score_field.png"
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"saved: {path}")


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    start = time.perf_counter()

    X_T, H, centers0, stats = simulate_2d_malliavin_ito(
        n_paths=300_000,
        T=0.35,
        n_steps=120,
        sigma=0.45,
        gamma_reg=1e-3,
        device=device,
    )

    print("simulation stats:", stats)
    print(f"simulation time: {time.perf_counter() - start:.2f}s")

    points, scores, counts = bin_teacher_2d(
        X_T,
        H,
        n_bins=80,
        min_count=30,
    )

    print(
        f"binned teacher points={points.shape[0]}, "
        f"count_mean={counts.mean().item():.1f}, "
        f"score_norm_mean={scores.norm(dim=1).mean().item():.3f}"
    )

    points, scores, counts = smooth_grid_scores(points, scores, counts)

    model = train_mlp(points, scores, counts, device=device)

    outdir = Path("results/2d_malliavin_binned_teacher")
    outdir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "points": points.detach().cpu(),
            "scores": scores.detach().cpu(),
            "counts": counts.detach().cpu(),
            "stats": stats,
        },
        outdir / "score_mlp_2d_binned.pt",
    )
    print(f"saved: {outdir / 'score_mlp_2d_binned.pt'}")

    plot_results(model, X_T, centers0, points, scores, counts, device=device)


if __name__ == "__main__":
    run()