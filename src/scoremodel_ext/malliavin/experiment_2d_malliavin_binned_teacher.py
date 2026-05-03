import math
import time
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


def jac_drift(x):
    n = x.shape[0]
    device = x.device
    I = torch.eye(2, device=device).expand(n, 2, 2)

    r2 = (x * x).sum(dim=1)
    outer = x[:, :, None] * x[:, None, :]

    return -(0.1 + 0.01 * r2)[:, None, None] * I - 0.02 * outer


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


def simulate_2d_malliavin_ito(
    n_paths=300_000,
    T=0.35,
    n_steps=120,
    sigma=0.45,
    gamma_reg=1e-3,
    device="cuda",
):
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)

    x, centers0 = sample_8gmm(n_paths, device=device)

    Y = torch.eye(2, device=device).expand(n_paths, 2, 2).clone()

    Y_list = []
    dW_list = []

    for _ in range(n_steps):
        Y_list.append(Y)

        dW = sqrt_dt * torch.randn(n_paths, 2, device=device)
        dW_list.append(dW)

        x_old = x
        Y_old = Y

        x = x_old + drift(x_old) * dt + sigma * dW

        J = jac_drift(x_old)
        Y = Y_old + torch.bmm(J, Y_old) * dt

    X_T = x
    Y_T = Y

    core = torch.zeros(n_paths, 2, 2, device=device)
    invY_list = []

    for Ys in Y_list:
        invYs = torch.linalg.inv(Ys)
        invY_list.append(invYs)
        core += sigma**2 * torch.bmm(invYs, invYs.transpose(1, 2)) * dt

    gamma = torch.bmm(torch.bmm(Y_T, core), Y_T.transpose(1, 2))

    eye = torch.eye(2, device=device).expand(n_paths, 2, 2)
    gamma_inv = torch.linalg.inv(gamma + gamma_reg * eye)

    delta = torch.zeros(n_paths, 2, device=device)

    for invYs, dW in zip(invY_list, dW_list):
        DsXT = sigma * torch.bmm(Y_T, invYs)
        U = torch.bmm(DsXT.transpose(1, 2), gamma_inv)
        delta += torch.bmm(U.transpose(1, 2), dW[:, :, None]).squeeze(-1)

    # Ito-only Malliavin teacher weight
    H = -delta

    stats = {
        "X_mean": X_T.mean(dim=0).detach().cpu(),
        "X_std": X_T.std(dim=0).detach().cpu(),
        "gamma_min_eig": torch.linalg.eigvalsh(gamma).min().item(),
        "gamma_trace_mean": gamma.diagonal(dim1=1, dim2=2).sum(dim=1).mean().item(),
        "H_norm_mean": H.norm(dim=1).mean().item(),
        "H_norm_std": H.norm(dim=1).std().item(),
    }

    return X_T, H, centers0, stats


def binned_teacher_2d(
    X,
    H,
    n_bins=80,
    min_count=30,
    q_low=0.005,
    q_high=0.995,
):
    device = X.device

    x_min, x_max = torch.quantile(X[:, 0], torch.tensor([q_low, q_high], device=device))
    y_min, y_max = torch.quantile(X[:, 1], torch.tensor([q_low, q_high], device=device))

    x_edges = torch.linspace(x_min, x_max, n_bins + 1, device=device)
    y_edges = torch.linspace(y_min, y_max, n_bins + 1, device=device)

    ix = torch.bucketize(X[:, 0], x_edges) - 1
    iy = torch.bucketize(X[:, 1], y_edges) - 1

    mask = (ix >= 0) & (ix < n_bins) & (iy >= 0) & (iy < n_bins)

    ix = ix[mask]
    iy = iy[mask]
    H = H[mask]

    flat = ix * n_bins + iy
    n_cells = n_bins * n_bins

    counts = torch.bincount(flat, minlength=n_cells).float()

    sum_h0 = torch.bincount(flat, weights=H[:, 0], minlength=n_cells)
    sum_h1 = torch.bincount(flat, weights=H[:, 1], minlength=n_cells)

    score0 = sum_h0 / counts.clamp_min(1.0)
    score1 = sum_h1 / counts.clamp_min(1.0)

    valid = counts >= min_count

    cell_ids = torch.arange(n_cells, device=device)
    valid_ids = cell_ids[valid]

    vx = valid_ids // n_bins
    vy = valid_ids % n_bins

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    points = torch.stack([x_centers[vx], y_centers[vy]], dim=1)
    scores = torch.stack([score0[valid], score1[valid]], dim=1)
    counts_valid = counts[valid]

    return points, scores, counts_valid, (x_edges, y_edges)


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
        T=1.0,
        n_steps=120,
        sigma=0.8,
        gamma_reg=1e-3,
        device=device,
    )

    print("simulation stats:", stats)
    print(f"simulation time: {time.perf_counter() - start:.2f}s")

    points, scores, counts, edges = binned_teacher_2d(
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