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
    # nonlinear radial mean-reverting drift:
    # b(x) = -x - ||x||^2 x
    r2 = (x * x).sum(dim=1, keepdim=True)
    return -x - r2 * x


def jac_drift(x):
    # Db(x) = -I - ||x||^2 I - 2 x x^T
    n = x.shape[0]
    device = x.device
    I = torch.eye(2, device=device).expand(n, 2, 2)

    r2 = (x * x).sum(dim=1)
    outer = x[:, :, None] * x[:, None, :]

    J = -(1.0 + r2)[:, None, None] * I - 2.0 * outer
    return J


class ScoreMLP(nn.Module):
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


def simulate_malliavin_batch(
    n_paths,
    T=1.0,
    n_steps=120,
    sigma=0.8,
    gamma_reg=1e-3,
    device="cuda",
):
    """
    Simulate nonlinear additive SDE:

        dX_t = b(X_t) dt + sigma dW_t

    together with first variation Y_t.

    Malliavin covariance:

        gamma_T = integral_0^T D_s X_T D_s X_T^T ds
                = Y_T [ integral_0^T Y_s^{-1} sigma^2 I Y_s^{-T} ds ] Y_T^T

    Ito part of Malliavin weight:

        delta_k^Ito = integral_0^T u_s^k · dW_s

    with

        u_s^k = (D_s X_T)^T gamma_T^{-1} e_k.

    Then noisy teacher:

        score_hat_i = - delta_i^Ito.

    This is NOT the full Skorokhod-corrected estimator yet.
    """
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)

    x, centers = sample_8gmm(n_paths, device=device)

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

    # compute core = integral Y_s^{-1} sigma^2 I Y_s^{-T} ds
    core = torch.zeros(n_paths, 2, 2, device=device)

    invY_list = []
    for Ys in Y_list:
        invYs = torch.linalg.inv(Ys)
        invY_list.append(invYs)
        core = core + sigma**2 * torch.bmm(invYs, invYs.transpose(1, 2)) * dt

    gamma = torch.bmm(torch.bmm(Y_T, core), Y_T.transpose(1, 2))
    eye = torch.eye(2, device=device).expand(n_paths, 2, 2)
    gamma_inv = torch.linalg.inv(gamma + gamma_reg * eye)

    # Ito Malliavin weight for each component k=1,2
    delta = torch.zeros(n_paths, 2, device=device)

    for invYs, dW in zip(invY_list, dW_list):
        # D_s X_T = Y_T Y_s^{-1} sigma
        DsXT = sigma * torch.bmm(Y_T, invYs)  # [N, 2, 2], columns for Brownian directions

        # u_s^k = (D_s X_T)^T gamma^{-1} e_k
        # U matrix columns k; rows Brownian dimensions.
        U = torch.bmm(DsXT.transpose(1, 2), gamma_inv)  # [N, 2, 2]

        # delta_k += u_s^k · dW_s
        delta = delta + torch.bmm(U.transpose(1, 2), dW[:, :, None]).squeeze(-1)

    teacher_score = -delta

    t = torch.full((n_paths,), T, device=device)

    stats = {
        "X_mean": X_T.mean(dim=0).detach().cpu(),
        "X_std": X_T.std(dim=0).detach().cpu(),
        "gamma_min_eig": torch.linalg.eigvalsh(gamma).min().item(),
        "gamma_mean_trace": gamma.diagonal(dim1=1, dim2=2).sum(dim=1).mean().item(),
        "teacher_mean_norm": teacher_score.norm(dim=1).mean().item(),
        "teacher_std_norm": teacher_score.norm(dim=1).std().item(),
    }

    return t, X_T, teacher_score, centers, stats


def train(
    n_train_steps=20_000,
    batch_size=4096,
    lr=2e-4,
    T=1.0,
    n_sde_steps=120,
    sigma=0.8,
    gamma_reg=1e-3,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    model = ScoreMLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    ema_loss = None

    for step in range(1, n_train_steps + 1):
        t, x, teacher, _, stats = simulate_malliavin_batch(
            batch_size,
            T=T,
            n_steps=n_sde_steps,
            sigma=sigma,
            gamma_reg=gamma_reg,
            device=device,
        )

        pred = model(t, x)
        loss = ((pred - teacher) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        ema_loss = loss.item() if ema_loss is None else 0.98 * ema_loss + 0.02 * loss.item()

        if step % 500 == 0:
            print(
                f"step={step:6d}, loss={loss.item():.6e}, ema={ema_loss:.6e}, "
                f"gamma_min={stats['gamma_min_eig']:.3e}, teacher_norm={stats['teacher_mean_norm']:.3e}"
            )

    return model


@torch.no_grad()
def plot_results(
    model,
    T=1.0,
    n_sde_steps=120,
    sigma=0.8,
    gamma_reg=1e-3,
    device="cuda",
):
    outdir = Path("results/2d_malliavin_teacher")
    outdir.mkdir(parents=True, exist_ok=True)

    # samples + teacher vectors
    t, x, teacher, centers, stats = simulate_malliavin_batch(
        12_000,
        T=T,
        n_steps=n_sde_steps,
        sigma=sigma,
        gamma_reg=gamma_reg,
        device=device,
    )

    pred = model(t, x)

    plt.figure(figsize=(7, 7))
    plt.scatter(
        x[:, 0].detach().cpu(),
        x[:, 1].detach().cpu(),
        s=2,
        alpha=0.25,
        label="X_T samples",
    )
    plt.scatter(
        centers[:, 0].detach().cpu(),
        centers[:, 1].detach().cpu(),
        s=80,
        marker="x",
        label="initial modes",
    )
    plt.xlim(-3.2, 3.2)
    plt.ylim(-3.2, 3.2)
    plt.gca().set_aspect("equal")
    plt.title("Nonlinear SDE terminal samples")
    plt.legend()
    plt.tight_layout()
    path = outdir / "terminal_samples.png"
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"saved: {path}")

    # score field learned by MLP
    grid_n = 35
    lim = 3.2
    xs = torch.linspace(-lim, lim, grid_n, device=device)
    ys = torch.linspace(-lim, lim, grid_n, device=device)
    Xg, Yg = torch.meshgrid(xs, ys, indexing="xy")
    points = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1)
    tg = torch.full((points.shape[0],), T, device=device)

    score = model(tg, points)

    pts = points.detach().cpu()
    sc = score.detach().cpu()

    plt.figure(figsize=(7, 7))
    plt.scatter(
        x[:, 0].detach().cpu(),
        x[:, 1].detach().cpu(),
        s=2,
        alpha=0.15,
        label="X_T samples",
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
        alpha=0.85,
    )
    plt.scatter(
        centers[:, 0].detach().cpu(),
        centers[:, 1].detach().cpu(),
        s=80,
        marker="x",
        label="initial modes",
    )
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.gca().set_aspect("equal")
    plt.title("Learned score field from Malliavin Ito teacher")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    path = outdir / "learned_score_field.png"
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"saved: {path}")

    # teacher vs prediction scatter for components
    err = (pred - teacher).detach().cpu()
    teacher_cpu = teacher.detach().cpu()
    pred_cpu = pred.detach().cpu()

    plt.figure(figsize=(7, 5))
    plt.scatter(teacher_cpu[:, 0], pred_cpu[:, 0], s=3, alpha=0.25, label="component 1")
    plt.scatter(teacher_cpu[:, 1], pred_cpu[:, 1], s=3, alpha=0.25, label="component 2")
    lo = min(teacher_cpu.min().item(), pred_cpu.min().item())
    hi = max(teacher_cpu.max().item(), pred_cpu.max().item())
    plt.plot([lo, hi], [lo, hi], linewidth=2, label="y=x")
    plt.xlabel("Malliavin teacher")
    plt.ylabel("MLP prediction")
    plt.title(f"Teacher vs prediction, RMSE={err.pow(2).mean().sqrt().item():.4f}")
    plt.legend()
    plt.tight_layout()
    path = outdir / "teacher_vs_prediction.png"
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"saved: {path}")

    print("stats:", stats)


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = train(device=device)

    outdir = Path("results/2d_malliavin_teacher")
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), outdir / "score_mlp_malliavin_ito.pt")
    print(f"saved: {outdir / 'score_mlp_malliavin_ito.pt'}")

    plot_results(model, device=device)


if __name__ == "__main__":
    run()