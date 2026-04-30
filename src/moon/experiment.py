import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
from sklearn.datasets import make_moons
import numpy as np
import os

device = "cpu"

# =========================================================
# 0. Matplotlib cache (avoid permission issues in containers)
# =========================================================
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# =========================================================
# 1. Moon data
# =========================================================
def get_data(n=10000, noise=0.05):
    X, _ = make_moons(n_samples=n, noise=noise)
    return torch.tensor(X, dtype=torch.float32)

# =========================================================
# 2. VE forward diffusion: x = x0 + sigma * eps
# Target score: s*(x, sigma) = -eps / sigma
# =========================================================
def q_sample_ve(x0, sigma):
    eps = torch.randn_like(x0)
    xt = x0 + sigma * eps
    target_score = -eps / sigma
    return xt, target_score

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

# =========================================================
# 4. Training: weighted DSM for VE
# loss = E[ sigma^2 * || s_theta(x0+sigma eps, sigma) - (-eps/sigma) ||^2 ]
# This stabilizes scales across sigmas.
# =========================================================
def train(model, data, epochs=3000, batch_size=1024, lr=1e-3,
          sigma_min=0.1, sigma_max=1.0, log_sigma=True):
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr)

    n = len(data)
    for step in range(epochs):
        idx = torch.randint(0, n, (batch_size,))
        x0 = data[idx]

        # sample sigma
        if log_sigma:
            # log-uniform is common for VE: sigma ~ LogUniform[sigma_min, sigma_max]
            u = torch.rand(batch_size, 1)
            sigma = sigma_min * (sigma_max / sigma_min) ** u
        else:
            sigma = torch.rand(batch_size, 1) * (sigma_max - sigma_min) + sigma_min

        xt, target = q_sample_ve(x0, sigma)
        pred = model(xt, sigma)

        # weighted DSM
        loss = (sigma**2 * (pred - target)**2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 200 == 0:
            print(f"step {step}, loss {loss.item():.4f}")

# =========================================================
# 5. Samplers
# =========================================================

@torch.no_grad()
def sample_ald(model, n=2000, sigma_min=0.1, sigma_max=1.0,
               n_levels=40, n_inner=50, step_size=0.02):
    """
    Annealed Langevin Dynamics (robust)
      x <- x + (eps/2) * score(x, sigma) + sqrt(eps) * z
    We anneal sigma from large to small.
    """
    model.eval()

    # sigma schedule: logspace is typical
    sigmas = torch.logspace(np.log10(sigma_max), np.log10(sigma_min), n_levels)

    # init at high noise scale
    x = torch.randn(n, 2) * float(sigmas[0])

    for sigma in sigmas:
        sigma_val = float(sigma)
        sigma_batch = torch.full((n, 1), sigma_val)

        # scale step size with sigma^2 (common heuristic; stabilizes across levels)
        eps = step_size * (sigma_val / float(sigmas[-1]))**2

        for _ in range(n_inner):
            score = model(x, sigma_batch)
            x = x + 0.5 * eps * score + np.sqrt(eps) * torch.randn_like(x)

    return x.detach()

@torch.no_grad()
def sample_pf_ode_simple(model, n=2000, sigma_min=0.1, sigma_max=1.0, steps=200):
    """
    Very simple deterministic flow using sigma schedule.
    NOTE: This is not the fully rigorous PF-ODE for VE;
    it's a baseline to compare after ALD works.
    """
    model.eval()
    sigmas = torch.linspace(sigma_max, sigma_min, steps)

    x = torch.randn(n, 2) * sigma_max

    for i in range(steps - 1):
        sigma = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])
        ds = sigma_next - sigma  # negative
        sigma_batch = torch.full((n, 1), sigma)
        score = model(x, sigma_batch)

        # heuristic drift: moves along score with sigma^2 scale
        x = x + (sigma**2) * score * ds

    return x.detach()

@torch.no_grad()
def sample_reverse_sde_simple(model, n=2000, sigma_min=0.1, sigma_max=1.0, steps=200):
    """
    Very simple stochastic sampler.
    NOTE: Not fully rigorous; used for qualitative comparison.
    """
    model.eval()
    sigmas = torch.linspace(sigma_max, sigma_min, steps)

    x = torch.randn(n, 2) * sigma_max

    for i in range(steps - 1):
        sigma = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])
        ds = sigma_next - sigma  # negative
        sigma_batch = torch.full((n, 1), sigma)
        score = model(x, sigma_batch)

        # heuristic: drift + diffusion
        x = x + (sigma**2) * score * ds + sigma * np.sqrt(abs(ds)) * torch.randn_like(x)

    return x.detach()

@torch.no_grad()
def sample_reverse_sde_correct(model, n=2000, steps=400,
                               sigma_max=1.0, sigma_min=0.1):

    model.eval()

    sigmas = torch.linspace(sigma_max, sigma_min, steps)
    x = torch.randn(n,2) * sigma_max

    for i in range(steps-1):
        sigma = float(sigmas[i])
        sigma_next = float(sigmas[i+1])
        ds = sigma_next - sigma  # negative

        sigma_batch = torch.full((n,1), sigma)
        score = model(x, sigma_batch)

        drift = - sigma * score * ds

        diffusion = np.sqrt(sigma**2 * abs(ds)) * torch.randn_like(x)

        x = x + drift + diffusion
    return x.detach()

@torch.no_grad()
def sample_pf_ode_correct(model, n=2000, sigma_min=0.1, sigma_max=1.0, steps=200):
    model.eval()

    sigmas = torch.linspace(sigma_max, sigma_min, steps)
    x = torch.randn(n,2) * sigma_max

    for i in range(steps-1):
        sigma = float(sigmas[i])
        sigma_next = float(sigmas[i+1])
        ds = sigma_next - sigma  # negative

        sigma_batch = torch.full((n,1), sigma)
        score = model(x, sigma_batch)

        # 正しい係数
        x = x - sigma * score * ds

    return x.detach()

# =========================================================
# 6. Plot utilities
# =========================================================
def scatter(ax, pts, title, s=5):
    ax.scatter(pts[:,0], pts[:,1], s=s)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")


# =========================================================
# 7. KDE utilities (for visualization)
# =========================================================
from sklearn.neighbors import KernelDensity
import plotly.graph_objects as go

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


# =========================================================
# 7. Run experiment
# =========================================================

if __name__ == "__main__":
    # --- settings (start with these) ---
    data_n = 5000
    data_noise = 0.05

    epochs = 4000          # Moonが出ない時はまずここを増やす
    batch_size = 1024
    lr = 1e-3

    sigma_min = 0.1
    sigma_max = 1.0
    log_sigma = True       # VEはlog-uniformが効きやすい

    sample_n = 2000

    # ALD params (robust)
    ald_levels = 40
    ald_inner = 60
    ald_step_size = 0.02

    # ODE/SDE baseline params
    ode_steps = 300
    sde_steps = 300

    out_png = "moon_result.png"

    print("Generating data...")
    data = get_data(data_n, noise=data_noise)

    print("Building model...")
    model = ScoreNet(hidden=256)

    print("Training...")
    train(
        model, data,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        log_sigma=log_sigma
    )

    print("Sampling (ALD)...")
    samples_ald = sample_ald(
        model,
        n=sample_n,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        n_levels=ald_levels,
        n_inner=ald_inner,
        step_size=ald_step_size
    )

    print("Sampling (PF-ODE simple)...")
    samples_ode = sample_pf_ode_correct(
        model,
        n=sample_n,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        steps=sde_steps
    )


    print("Sampling (Reverse SDE simple)...")
    samples_sde = sample_reverse_sde_correct(
        model,
        n=sample_n,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        steps=sde_steps
    )

    print("Plotting...")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    scatter(axes[0], data.numpy(), "True Moon")
    scatter(axes[1], samples_ald.numpy(), "ALD (robust)")
    scatter(axes[2], samples_ode.numpy(), "ODE (simple baseline)")
    scatter(axes[3], samples_sde.numpy(), "SDE (simple baseline)")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"Saved {out_png}")