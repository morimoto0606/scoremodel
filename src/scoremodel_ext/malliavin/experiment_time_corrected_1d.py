import time
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")

from scoremodel_ext.malliavin.experiment_nonlinear_pde1d import simulate_malliavin
from scoremodel_ext.malliavin.models import TimeScoreMLP


def smooth_1d_score(score, counts, radius=2, eps=1e-8):
    """
    Count-weighted local smoothing over neighboring bins.
    """
    if radius <= 0:
        return score

    smoothed = torch.zeros_like(score)
    weight_sum = torch.zeros_like(score)

    for shift in range(-radius, radius + 1):
        if shift < 0:
            src = score[-shift:]
            w = counts[-shift:]
            smoothed[:shift] += src * w
            weight_sum[:shift] += w
        elif shift > 0:
            src = score[:-shift]
            w = counts[:-shift]
            smoothed[shift:] += src * w
            weight_sum[shift:] += w
        else:
            smoothed += score * counts
            weight_sum += counts

    return smoothed / weight_sum.clamp_min(eps)


def make_teacher_dataset(
    times=(0.2, 0.4, 0.6, 0.8, 1.0),
    n_paths=200_000,
    n_steps=300,
    sigma=0.8,
    x0=1.5,
    n_bins=160,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    all_t = []
    all_y = []
    all_score = []
    all_counts = []

    print("building corrected Malliavin teacher dataset...")
    total_start = time.perf_counter()
    for T in times:
        start = time.perf_counter()
        mc = simulate_malliavin(
            n_paths=n_paths,
            n_steps=n_steps,
            T=float(T),
            sigma=sigma,
            x0=x0,
            n_bins=n_bins,
            device=device,
        )

        elapsed = time.perf_counter() - start
        y = torch.tensor(mc["centers"], dtype=torch.float32, device=device)
        score = torch.tensor(mc["score_corr"], dtype=torch.float32, device=device)
        counts = torch.tensor(mc["counts"], dtype=torch.float32, device=device)
        score = smooth_1d_score(score, counts, radius=2, eps=1e-8)
        t = torch.full_like(y, float(T))

        all_t.append(t)
        all_y.append(y)
        all_score.append(score)
        all_counts.append(counts)

        print(
            f"T={T:.2f}: bins={len(y):4d}, "
            f"time={elapsed:.2f}s, "
            f"score_range=({score.min().item():.3f}, {score.max().item():.3f}), "
            f"count_mean={counts.float().mean().item():.1f}"
        )

    total_elapsed = time.perf_counter() - total_start
    print(f"teacher dataset total time: {total_elapsed:.2f}s")

    t = torch.cat(all_t)
    y = torch.cat(all_y)
    score = torch.cat(all_score)
    counts = torch.cat(all_counts)

    return t, y, score, counts


def train_time_score_mlp(
    t,
    y,
    score,
    counts,
    n_epochs=8000,
    batch_size=512,
    lr=1e-3,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = TimeScoreMLP(hidden=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    weights = counts / counts.mean()
    n = y.shape[0]

    best_loss = float("inf")
    best_state = None

    print("training time-conditioned score MLP...")
    start = time.perf_counter()

    for epoch in range(1, n_epochs + 1):
        idx = torch.randint(0, n, (batch_size,), device=device)

        pred = model(t[idx], y[idx]).squeeze(-1)
        loss = (weights[idx] * (pred - score[idx]) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if epoch % 500 == 0:
            with torch.no_grad():
                full_pred = model(t, y).squeeze(-1)
                full_loss = (weights * (full_pred - score) ** 2).mean()

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
        print(f"loaded best checkpoint: {best_loss:.6e}")

    total = time.perf_counter() - start
    print(f"training total time: {total:.2f}s")
    return model


@torch.no_grad()
def plot_time_fits(model, t, y, score, counts):
    outdir = Path("results/malliavin_time_corrected_1d")
    outdir.mkdir(parents=True, exist_ok=True)

    unique_times = torch.unique(t).detach().cpu().numpy()

    for T in unique_times:
        mask = torch.isclose(t, torch.tensor(float(T), device=t.device))
        y_T = y[mask]
        score_T = score[mask]
        counts_T = counts[mask]

        order = torch.argsort(y_T)
        y_T = y_T[order]
        score_T = score_T[order]
        counts_T = counts_T[order]

        tt = torch.full_like(y_T, float(T))
        pred_T = model(tt, y_T).squeeze(-1)

        plt.figure(figsize=(8, 5))
        sc = plt.scatter(
            y_T.detach().cpu(),
            score_T.detach().cpu(),
            c=counts_T.detach().cpu(),
            cmap="viridis",
            s=18,
            alpha=0.8,
            label="corrected teacher",
        )
        plt.colorbar(sc, label="bin count")
        plt.plot(
            y_T.detach().cpu(),
            pred_T.detach().cpu(),
            linewidth=2.2,
            label="MLP",
        )
        plt.xlabel("y")
        plt.ylabel("score")
        plt.title(f"Time-conditioned corrected score, T={float(T):.2f}")
        plt.legend()
        plt.tight_layout()

        path = outdir / f"fit_T_{float(T):.2f}.png"
        plt.savefig(path, dpi=220)
        plt.close()
        print(f"saved: {path}")

    # one combined plot
    plt.figure(figsize=(8, 5))
    for T in unique_times:
        mask = torch.isclose(t, torch.tensor(float(T), device=t.device))
        y_T = y[mask]
        order = torch.argsort(y_T)
        y_T = y_T[order]
        tt = torch.full_like(y_T, float(T))
        pred_T = model(tt, y_T).squeeze(-1)

        plt.plot(
            y_T.detach().cpu(),
            pred_T.detach().cpu(),
            linewidth=2,
            label=f"T={float(T):.2f}",
        )

    plt.xlabel("y")
    plt.ylabel("learned score")
    plt.title("Learned corrected score across time")
    plt.legend()
    plt.tight_layout()
    path = outdir / "learned_scores_all_times.png"
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"saved: {path}")


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    # add low-time teachers for better reverse sampling near t=0
    times = (0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00)

    t, y, score, counts = make_teacher_dataset(
        times=times,
        n_paths=300_000,
        n_steps=400,
        sigma=0.8,
        x0=1.5,
        n_bins=180,
        device=device,
    )

    model = train_time_score_mlp(
        t,
        y,
        score,
        counts,
        n_epochs=12000,
        batch_size=512,
        lr=1e-3,
        device=device,
    )

    outdir = Path("results/malliavin_time_corrected_1d")
    outdir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden": 128,
            "t": t.detach().cpu(),
            "y": y.detach().cpu(),
            "score": score.detach().cpu(),
            "counts": counts.detach().cpu(),
            "times": times,
        },
        outdir / "time_score_mlp.pt",
    )
    print(f"saved: {outdir / 'time_score_mlp.pt'}")

    plot_time_fits(model, t, y, score, counts)


if __name__ == "__main__":
    run()