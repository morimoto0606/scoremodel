import math

import torch


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


@torch.no_grad()
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


@torch.no_grad()
def bin_teacher_2d(
    X,
    H,
    n_bins=80,
    min_count=25,
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

    sum0 = torch.bincount(flat, weights=H[:, 0], minlength=n_cells)
    sum1 = torch.bincount(flat, weights=H[:, 1], minlength=n_cells)

    s0 = sum0 / counts.clamp_min(1.0)
    s1 = sum1 / counts.clamp_min(1.0)

    valid = counts >= min_count
    ids = torch.arange(n_cells, device=device)[valid]

    vx = ids // n_bins
    vy = ids % n_bins

    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    yc = 0.5 * (y_edges[:-1] + y_edges[1:])

    pts = torch.stack([xc[vx], yc[vy]], dim=1)
    sc = torch.stack([s0[valid], s1[valid]], dim=1)
    cc = counts[valid]

    return pts, sc, cc
