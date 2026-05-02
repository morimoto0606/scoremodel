import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use("seaborn-v0_8")

from scoremodel_ext.malliavin.experiment_nonlinear_pde1d import simulate_malliavin


class ScoreMLP(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, y):
        return self.net(y)


def run(
    n_epochs=4000,
    batch_size=512,
    lr=1e-3,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    mc = simulate_malliavin(
        n_paths=300_000,
        n_steps=400,
        T=1.0,
        sigma=0.8,
        x0=1.5,
        n_bins=180,
        device=device,
    )

    y = torch.tensor(mc["centers"], dtype=torch.float32, device=device)[:, None]
    score = torch.tensor(mc["score_corr"], dtype=torch.float32, device=device)[:, None]
    counts = torch.tensor(mc["counts"], dtype=torch.float32, device=device)[:, None]

    # count が多い bin を少し重くする
    weights = counts / counts.mean()

    model = ScoreMLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    n = y.shape[0]

    for epoch in range(1, n_epochs + 1):
        idx = torch.randint(0, n, (batch_size,), device=device)

        pred = model(y[idx])
        loss = (weights[idx] * (pred - score[idx]) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 500 == 0:
            with torch.no_grad():
                full_loss = (weights * (model(y) - score) ** 2).mean()
            print(f"epoch={epoch:5d}, loss={full_loss.item():.6e}")

    outdir = Path("results/malliavin_teacher_mlp_1d")
    outdir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        y_plot = torch.linspace(y.min(), y.max(), 500, device=device)[:, None]
        pred_plot = model(y_plot)

    plt.figure(figsize=(8, 5))
    plt.scatter(
        y.detach().cpu().numpy(),
        score.detach().cpu().numpy(),
        s=14,
        alpha=0.65,
        label="Malliavin corrected teacher",
    )
    plt.plot(
        y_plot.detach().cpu().numpy(),
        pred_plot.detach().cpu().numpy(),
        linewidth=2,
        label="MLP learned score",
    )
    plt.xlabel("y")
    plt.ylabel("score")
    plt.title("MLP learning Malliavin-corrected score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "mlp_fit_score.png", dpi=220)
    plt.close()

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "centers": mc["centers"],
            "score_corr": mc["score_corr"],
            "counts": mc["counts"],
        },
        outdir / "teacher_mlp.pt",
    )

    print(f"saved: {outdir / 'mlp_fit_score.png'}")
    print(f"saved: {outdir / 'teacher_mlp.pt'}")


if __name__ == "__main__":
    run()