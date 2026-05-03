import time
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")

try:
    from .models import TimeScoreMLP2D
    from .sde_2d import simulate_2d_malliavin_ito, bin_teacher_2d
except ImportError:
    from models import TimeScoreMLP2D
    from sde_2d import simulate_2d_malliavin_ito, bin_teacher_2d


# =====================================================
# build all times
# =====================================================

def build_dataset(
    times,
    device="cuda",
):
    all_t = []
    all_x = []
    all_s = []
    all_c = []

    total_start = time.perf_counter()

    for T in times:
        st = time.perf_counter()

        X, H, _, _ = simulate_2d_malliavin_ito(
            T=T,
            n_paths=250_000,
            n_steps=120,
            sigma=0.45,
            gamma_reg=1e-3,
            device=device,
        )

        pts, sc, cc = bin_teacher_2d(X, H)

        tt = torch.full((pts.shape[0],), float(T), device=device)

        all_t.append(tt)
        all_x.append(pts)
        all_s.append(sc)
        all_c.append(cc)

        print(
            f"T={T:.2f}: points={pts.shape[0]:4d}, "
            f"score_norm={sc.norm(dim=1).mean().item():.3f}, "
            f"time={time.perf_counter()-st:.1f}s"
        )

    print(f"dataset total time: {time.perf_counter()-total_start:.1f}s")

    return (
        torch.cat(all_t),
        torch.cat(all_x),
        torch.cat(all_s),
        torch.cat(all_c),
    )


# =====================================================
# train
# =====================================================

def train_model(
    t,
    x,
    s,
    c,
    n_epochs=12000,
    batch_size=2048,
    lr=2e-4,
    device="cuda",
):
    model = TimeScoreMLP2D().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    w = c / c.mean()
    n = x.shape[0]

    best = float("inf")
    best_state = None

    start = time.perf_counter()

    for ep in range(1, n_epochs + 1):
        idx = torch.randint(0, n, (batch_size,), device=device)

        pred = model(t[idx], x[idx])
        loss = (w[idx, None] * (pred - s[idx]) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if ep % 500 == 0:
            with torch.no_grad():
                full = model(t, x)
                full_loss = (w[:, None] * (full - s) ** 2).mean()

            if full_loss.item() < best:
                best = full_loss.item()
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                print(f"*** best updated: {best:.6e}")

            print(
                f"epoch={ep:5d}, "
                f"loss={full_loss.item():.6e}, "
                f"best={best:.6e}"
            )

    model.load_state_dict(best_state)
    return model


# =====================================================
# plot
# =====================================================

@torch.no_grad()
def plot_snapshot(model, T, path, device="cuda"):
    lim = 3.2
    n = 40

    xs = torch.linspace(-lim, lim, n, device=device)
    ys = torch.linspace(-lim, lim, n, device=device)

    Xg, Yg = torch.meshgrid(xs, ys, indexing="xy")
    grid = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1)

    tt = torch.full((grid.shape[0],), T, device=device)

    pred = model(tt, grid)

    plt.figure(figsize=(7, 7))
    plt.quiver(
        grid[:, 0].cpu(),
        grid[:, 1].cpu(),
        pred[:, 0].cpu(),
        pred[:, 1].cpu(),
        angles="xy",
        scale_units="xy",
        scale=22,
        width=0.002,
    )
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.gca().set_aspect("equal")
    plt.title(f"Learned score field T={T:.2f}")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"saved: {path}")


# =====================================================
# run
# =====================================================

def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    times = [0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00]

    t, x, s, c = build_dataset(times, device=device)

    model = train_model(t, x, s, c, device=device)

    outdir = Path("results/2d_time_malliavin_binned")
    outdir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "times": times,
        },
        outdir / "time_score_mlp_2d.pt",
    )

    print(f"saved: {outdir / 'time_score_mlp_2d.pt'}")

    for T in times:
        plot_snapshot(
            model,
            T,
            outdir / f"score_field_T_{T:.2f}.png",
            device=device,
        )


if __name__ == "__main__":
    run()