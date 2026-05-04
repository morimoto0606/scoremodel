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


def sigma_schedule(t, T, sigma=0.45, sigma_min=None, sigma_max=None):
    """
    If sigma_min/sigma_max are None, use constant sigma.
    Otherwise use linear schedule:
        sigma(t) = sigma_min + (sigma_max - sigma_min) * t/T
    """
    if sigma_min is None or sigma_max is None:
        return sigma

    return sigma_min + (sigma_max - sigma_min) * (t / T)


@torch.no_grad()
def simulate_2d_malliavin_ito(
    n_paths=300_000,
    T=0.35,
    n_steps=120,
    sigma=0.45,
    sigma_min=None,
    sigma_max=None,
    gamma_reg=1e-3,
    device="cuda",
):
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)

    x, centers0 = sample_8gmm(n_paths, device=device)

    Y = torch.eye(2, device=device).expand(n_paths, 2, 2).clone()

    Y_list = []
    dW_list = []
    sigma_list = []

    # forward simulation
    for k in range(n_steps):
        t_mid = (k + 0.5) * dt
        sigma_k = sigma_schedule(
            t_mid,
            T,
            sigma=sigma,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )

        Y_list.append(Y)
        sigma_list.append(sigma_k)

        dW = sqrt_dt * torch.randn(n_paths, 2, device=device)
        dW_list.append(dW)

        x_old = x
        Y_old = Y

        x = x_old + drift(x_old) * dt + sigma_k * dW

        J = jac_drift(x_old)
        Y = Y_old + torch.bmm(J, Y_old) * dt

    X_T = x
    Y_T = Y

    # Malliavin covariance
    core = torch.zeros(n_paths, 2, 2, device=device)
    invY_list = []

    for Ys, sigma_k in zip(Y_list, sigma_list):
        invYs = torch.linalg.inv(Ys)
        invY_list.append(invYs)

        core += (
            sigma_k**2
            * torch.bmm(invYs, invYs.transpose(1, 2))
            * dt
        )

    gamma = torch.bmm(torch.bmm(Y_T, core), Y_T.transpose(1, 2))

    eye = torch.eye(2, device=device).expand(n_paths, 2, 2)
    gamma_inv = torch.linalg.inv(gamma + gamma_reg * eye)

    # Ito Malliavin weight
    delta = torch.zeros(n_paths, 2, device=device)

    for invYs, dW, sigma_k in zip(invY_list, dW_list, sigma_list):
        # D_s X_T = Y_T Y_s^{-1} sigma_s
        DsXT = sigma_k * torch.bmm(Y_T, invYs)

        # U_s = (D_s X_T)^T gamma^{-1}
        U = torch.bmm(DsXT.transpose(1, 2), gamma_inv)

        delta += torch.bmm(U.transpose(1, 2), dW[:, :, None]).squeeze(-1)

    H = -delta

    eigvals = torch.linalg.eigvalsh(gamma)

    stats = {
        "X_mean": X_T.mean(dim=0).detach().cpu(),
        "X_std": X_T.std(dim=0).detach().cpu(),
        "gamma_min_eig": eigvals.min().item(),
        "gamma_trace_mean": gamma.diagonal(dim1=1, dim2=2).sum(dim=1).mean().item(),
        "H_norm_mean": H.norm(dim=1).mean().item(),
        "H_norm_std": H.norm(dim=1).std().item(),
        "sigma_min": sigma_min if sigma_min is not None else sigma,
        "sigma_max": sigma_max if sigma_max is not None else sigma,
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


@torch.no_grad()
def nw_teacher_2d(X_T, H, query_x, bandwidth=None, batch_size=64):
    """
    Nadaraya-Watson Gaussian kernel regression for 2D score.

    X_T      : (n, 2)  — forward-terminal samples
    H        : (n, 2)  — per-path Malliavin weights
    query_x  : (m, 2)  — query positions
    bandwidth: float or None (Silverman's rule if None)

    Returns score estimates at query_x: (m, 2)
    """
    device = X_T.device
    n = X_T.shape[0]

    if bandwidth is None:
        # Silverman's rule for d=2: h = n^{-1/(d+4)} * sigma_mean
        sigma_mean = float(X_T.std(dim=0).mean())
        h = sigma_mean * float(n) ** (-1.0 / 6.0)
    else:
        h = float(bandwidth)

    nq = query_x.shape[0]
    score = torch.zeros(nq, 2, device=device)

    for i in range(0, nq, batch_size):
        xq = query_x[i : i + batch_size]                      # (b, 2)
        diff = (xq[:, None, :] - X_T[None, :, :]) / h         # (b, n, 2)
        kw = torch.exp(-0.5 * (diff * diff).sum(-1))           # (b, n)
        denom = kw.sum(-1, keepdim=True).clamp_min(1e-12)      # (b, 1)
        score[i : i + batch_size] = (kw[:, :, None] * H[None, :, :]).sum(1) / denom

    return score


@torch.no_grad()
def knn_nw_teacher_2d(X_T, H, query_x, k=500, bandwidth_scale=1.0, batch_size=64):
    """
    kNN-adaptive Nadaraya-Watson kernel regression for 2D score.

    Bandwidth for each query point = bandwidth_scale * distance to k-th nearest
    training point.

    X_T            : (n, 2)
    H              : (n, 2)
    query_x        : (m, 2)
    k              : neighbourhood size
    bandwidth_scale: scalar multiplier applied to the kNN distance (default 1.0)

    Returns score estimates at query_x: (m, 2)
    """
    device = X_T.device
    k = min(k, X_T.shape[0])

    nq = query_x.shape[0]
    score = torch.zeros(nq, 2, device=device)

    for i in range(0, nq, batch_size):
        xq = query_x[i : i + batch_size]                      # (b, 2)
        diff = xq[:, None, :] - X_T[None, :, :]               # (b, n, 2)
        dist2 = (diff * diff).sum(-1)                          # (b, n)
        dist = dist2.sqrt()                                    # (b, n)
        h = (dist.kthvalue(k, dim=1).values * bandwidth_scale).clamp_min(1e-8)  # (b,)
        kw = torch.exp(-0.5 * (dist / h[:, None]) ** 2)       # (b, n)
        denom = kw.sum(-1, keepdim=True).clamp_min(1e-12)
        score[i : i + batch_size] = (kw[:, :, None] * H[None, :, :]).sum(1) / denom

    return score
