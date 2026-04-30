import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.neighbors import KernelDensity
import plotly.graph_objects as go

plt.style.use("ggplot")
device = "cpu"

# ======================
# 1. Data
# ======================
def get_data(n=5000):
    X, _ = make_moons(n_samples=n, noise=0.05)
    return torch.tensor(X, dtype=torch.float32)

# ======================
# 2. Forward (VE style)
# ======================
def q_sample(x0, sigma):
    noise = torch.randn_like(x0)
    xt = x0 + sigma * noise
    target = -noise / sigma
    return xt, target

# =========================================================
# 3. Score network: input (x, sigma) -> score in R^2
# =========================================================
class ScoreNet(nn.Module):
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

    def forward(self, x, sigma):
        return self.net(torch.cat([x, sigma], dim=1))


# ======================
# 4. Training
# ======================
def train(model, data, epochs=4000):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for i in range(epochs):
        sigma = torch.rand(len(data),1)*0.9 + 0.1
        xt, target = q_sample(data, sigma)
        pred = model(xt, sigma)

        loss = (sigma**2 * (pred - target)**2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 200 == 0:
            print(f"step {i}, loss {loss.item():.4f}")

# ======================
# 5. ODE sampler (Probability Flow)
# ======================
@torch.no_grad()
def sample_pf_ode(model, n=2000, steps=400,
                  sigma_max=1.0, sigma_min=0.1):

    sigmas = torch.linspace(sigma_max, sigma_min, steps)
    x = torch.randn(n,2) * sigma_max

    for i in range(steps-1):
        sigma = float(sigmas[i])
        sigma_next = float(sigmas[i+1])
        ds = sigma_next - sigma

        sigma_batch = torch.full((n,1), sigma)
        score = model(x, sigma_batch)

        x = x - sigma * score * ds

    return x.detach()

# ======================
# 6. Reverse SDE
# ======================
@torch.no_grad()
def sample_reverse_sde(model, n=2000, steps=400,
                       sigma_max=1.0, sigma_min=0.1):

    sigmas = torch.linspace(sigma_max, sigma_min, steps)
    x = torch.randn(n,2) * sigma_max

    for i in range(steps-1):
        sigma = float(sigmas[i])
        sigma_next = float(sigmas[i+1])
        ds = sigma_next - sigma

        sigma_batch = torch.full((n,1), sigma)
        score = model(x, sigma_batch)

        drift = - sigma * score * ds
        diffusion = np.sqrt(sigma**2 * abs(ds)) * torch.randn_like(x)

        x = x + drift + diffusion

    return x.detach()

# ======================
# 7. ALD sampler
# ======================
@torch.no_grad()
def sample_ald(model, n=2000, steps=50,
               sigma_max=1.0, sigma_min=0.1,
               inner_steps=10, step_size=0.01):

    sigmas = torch.linspace(sigma_max, sigma_min, steps)
    x = torch.randn(n,2) * sigma_max

    for sigma in sigmas:
        sigma_val = float(sigma)
        sigma_batch = torch.full((n,1), sigma_val)

        for _ in range(inner_steps):
            score = model(x, sigma_batch)
            x = x + step_size * score \
                  + np.sqrt(2*step_size) * torch.randn_like(x)

    return x.detach()

# ======================
# 8. KDE
# ======================
def compute_kde(data, grid_lim=3, grid_size=100, bandwidth=0.2):
    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(data)

    x = np.linspace(-grid_lim, grid_lim, grid_size)
    y = np.linspace(-grid_lim, grid_lim, grid_size)
    X, Y = np.meshgrid(x, y)
    XY = np.vstack([X.ravel(), Y.ravel()]).T

    Z = np.exp(kde.score_samples(XY))
    Z = Z.reshape(X.shape)

    return X, Y, Z

def plot_3d(X, Y, Z, title):

    # ① グリッドを間引く（超重要）
    stride = 2   # 2〜3にするとかなり軽くなる
    Xs = X[::stride, ::stride]
    Ys = Y[::stride, ::stride]
    Zs = Z[::stride, ::stride]

    fig = go.Figure(
        data=[
            go.Surface(
                x=Xs,
                y=Ys,
                z=Zs,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Density")
            )
        ]
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="Density"
        ),
        width=650,
        height=550
    )

    fig.show()

# ======================
# 10. Compare all
# ======================
def compare_all_methods(model, n=2000, steps=400):

    true_data = get_data(n).numpy()
    ode_data = sample_pf_ode(model, n=n, steps=steps).numpy()
    sde_data = sample_reverse_sde(model, n=n, steps=steps).numpy()
    ald_data = sample_ald(model, n=n).numpy()

    print("Plotting...")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    scatter(axes[0], true_data, "True Moon")
    scatter(axes[1], ald_data, "ALD (robust)")
    scatter(axes[2], ode_data, "ODE (simple baseline)")
    scatter(axes[3], sde_data, "SDE (simple baseline)")
    out_png = "moon_result2.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"Saved {out_png}")
    X, Y, Z_true = compute_kde(true_data)
    _, _, Z_ode = compute_kde(ode_data)
    _, _, Z_sde = compute_kde(sde_data)
    _, _, Z_ald = compute_kde(ald_data)

    vmin = min(Z_true.min(), Z_ode.min(), Z_sde.min(), Z_ald.min())
    vmax = max(Z_true.max(), Z_ode.max(), Z_sde.max(), Z_ald.max())

    plt.figure(figsize=(10,8))

    titles = ["True", "ODE", "SDE", "ALD"]
    Zs = [Z_true, Z_ode, Z_sde, Z_ald]

    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.imshow(Zs[i], origin="lower",
                   extent=[-3,3,-3,3],
                   vmin=vmin, vmax=vmax,
                   cmap="viridis")
        plt.title(titles[i])

    plt.tight_layout()
    plt.savefig("all_methods_heatmap.png", dpi=200)
    print("Saved all_methods_heatmap.png")

    # Difference plots
    plt.figure(figsize=(12,4))
    diffs = [Z_ode-Z_true, Z_sde-Z_true, Z_ald-Z_true]
    diff_titles = ["ODE-True", "SDE-True", "ALD-True"]

    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.imshow(diffs[i], origin="lower",
                   extent=[-3,3,-3,3],
                   cmap="coolwarm")
        plt.title(diff_titles[i])

    plt.tight_layout()
    plt.savefig("density_diff.png", dpi=200)
    print("Saved density_diff.png")

    plot_3d(X, Y, Z_true, "True Density")
    plot_3d(X, Y, Z_ode, "ODE Density")
    plot_3d(X, Y, Z_sde, "SDE Density")
    plot_3d(X, Y, Z_ald, "ALD Density")

# =========================================================
# 6. Plot utilities
# =========================================================
def scatter(ax, pts, title, s=5):
    ax.scatter(pts[:,0], pts[:,1], s=s)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")

# ======================
# 11. Main
# ======================
if __name__ == "__main__":

    data = get_data(5000)
    model = ScoreNet()

    print("Training...")
    train(model, data)

    compare_all_methods(model)


