"""
Nonlinear SDE from Mirafzali et al., Appendix C.

    dX_t = -k β(t) (X_t − a) / (1 + (X_t − a)²) dt + σ √β(t) dW_t

The drift acts **component-wise** on R² data; the Jacobian is therefore
diagonal, which the Malliavin weight computation exploits.

β schedule: linear   β(t) = β_min + (β_max − β_min) · t / T

Default configuration (Mirafzali Appendix C):
    k=1, σ=1, a=0, β_min=1, β_max=25, T=1, dt=0.004 (n_steps=250)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

import torch


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NonlinearSDEConfig:
    """Parameters for the Mirafzali nonlinear SDE."""
    k:        float = 1.0
    sigma:    float = 1.0
    a:        float = 0.0
    beta_min: float = 1.0
    beta_max: float = 25.0
    T:        float = 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Schedule helpers
# ──────────────────────────────────────────────────────────────────────────────

def beta_at(t: float, cfg: NonlinearSDEConfig) -> float:
    """Linear β schedule: β(t) = β_min + (β_max − β_min) · t / T."""
    return cfg.beta_min + (cfg.beta_max - cfg.beta_min) * (t / cfg.T)


def sigma_t(t: float, cfg: NonlinearSDEConfig) -> float:
    """Diffusion coefficient g(t) = σ √β(t)."""
    return cfg.sigma * math.sqrt(beta_at(t, cfg))


# ──────────────────────────────────────────────────────────────────────────────
# Drift and Jacobian
# ──────────────────────────────────────────────────────────────────────────────

def drift_nl(x: torch.Tensor, t: float, cfg: NonlinearSDEConfig) -> torch.Tensor:
    """
    Forward drift  b(x, t) = -k β(t) (x − a) / (1 + (x − a)²)

    Applied component-wise.

    Parameters
    ----------
    x : (n, 2)
    t : float

    Returns
    -------
    (n, 2) drift tensor
    """
    beta = beta_at(t, cfg)
    u = x - cfg.a
    return -cfg.k * beta * u / (1.0 + u * u)


def jac_drift_nl(x: torch.Tensor, t: float, cfg: NonlinearSDEConfig) -> torch.Tensor:
    """
    Diagonal Jacobian of the drift:

        ∂b_i/∂x_i = −k β(t) (1 − (x_i − a)²) / (1 + (x_i − a)²)²

    Off-diagonal entries are zero (component-wise drift).

    Parameters
    ----------
    x : (n, 2)

    Returns
    -------
    J : (n, 2, 2) with only J[:, 0, 0] and J[:, 1, 1] nonzero
    """
    beta = beta_at(t, cfg)
    u    = x - cfg.a                        # (n, 2)
    u2   = u * u                             # (n, 2)
    denom = (1.0 + u2) ** 2                 # (n, 2)
    diag = -cfg.k * beta * (1.0 - u2) / denom   # (n, 2)

    n = x.shape[0]
    J = torch.zeros(n, 2, 2, device=x.device, dtype=x.dtype)
    J[:, 0, 0] = diag[:, 0]
    J[:, 1, 1] = diag[:, 1]
    return J


# ──────────────────────────────────────────────────────────────────────────────
# Forward simulation — positions only (cheap; used for reverse starting points)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def simulate_forward_nl(
    X0: torch.Tensor,
    T: float,
    cfg: NonlinearSDEConfig,
    n_steps: int = 250,
) -> torch.Tensor:
    """
    Euler–Maruyama forward simulation without Malliavin weight tracking.

    Parameters
    ----------
    X0      : (n, 2) initial samples from p_data
    T       : terminal time
    n_steps : Euler–Maruyama discretisation steps

    Returns
    -------
    X_T : (n, 2) terminal positions
    """
    dt      = T / n_steps
    sqrt_dt = math.sqrt(dt)
    x       = X0.clone()

    for k in range(n_steps):
        t_mid   = (k + 0.5) * dt
        g       = sigma_t(t_mid, cfg)
        dW      = sqrt_dt * torch.randn_like(x)
        x       = x + drift_nl(x, t_mid, cfg) * dt + g * dW

    return x


# ──────────────────────────────────────────────────────────────────────────────
# Forward simulation + Malliavin weights
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def simulate_malliavin_nl(
    X0: torch.Tensor,
    T: float,
    cfg: NonlinearSDEConfig,
    n_steps: int = 250,
    gamma_reg: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Euler–Maruyama forward simulation with the Itô–Malliavin score weight.

    Algorithm (same structure as `simulate_2d_malliavin_ito` in sde_2d.py):

    1. Track the flow matrix Y_t (dY = J(X_t,t) Y dt, Y_0 = I).
    2. Build Malliavin covariance
           γ = Y_T (∫_0^T Y_s⁻¹ g²(s) (Y_s⁻¹)ᵀ ds) Y_Tᵀ
    3. Compute the Skorokhod integral δ = Σ_s U_sᵀ dW_s where
           U_s = (D_s X_T)ᵀ γ⁻¹  and  D_s X_T = g(s) Y_T Y_s⁻¹
    4. Return H = −δ  (Malliavin score weight).

    Parameters
    ----------
    X0        : (n, 2)
    T         : terminal time
    n_steps   : Euler steps
    gamma_reg : Tikhonov regularisation for γ inversion

    Returns
    -------
    X_T : (n, 2) terminal positions
    H   : (n, 2) Malliavin score weights  s.t.  ∇_x log p_T ≈ E[H | X_T = x]
    """
    n      = X0.shape[0]
    device = X0.device
    dtype  = X0.dtype
    dt     = T / n_steps
    sqrt_dt = math.sqrt(dt)

    x = X0.clone()
    Y = torch.eye(2, device=device, dtype=dtype).expand(n, 2, 2).clone()

    Y_list:      list = []
    dW_list:     list = []
    g_list:      list = []   # g(t) = σ √β(t) at each step midpoint

    # ── forward pass ──────────────────────────────────────────────────────────
    for k in range(n_steps):
        t_mid = (k + 0.5) * dt
        g     = sigma_t(t_mid, cfg)

        Y_list.append(Y)
        g_list.append(g)

        dW = sqrt_dt * torch.randn(n, 2, device=device, dtype=dtype)
        dW_list.append(dW)

        # Euler step for X
        x_new = x + drift_nl(x, t_mid, cfg) * dt + g * dW

        # Euler step for flow Y
        J = jac_drift_nl(x, t_mid, cfg)     # (n, 2, 2)
        Y = Y + torch.bmm(J, Y) * dt

        x = x_new

    X_T = x
    Y_T = Y

    # ── Malliavin covariance γ ─────────────────────────────────────────────
    core      = torch.zeros(n, 2, 2, device=device, dtype=dtype)
    invY_list = []

    for Ys, g in zip(Y_list, g_list):
        invYs = torch.linalg.inv(Ys)
        invY_list.append(invYs)
        core += g**2 * torch.bmm(invYs, invYs.transpose(1, 2)) * dt

    gamma = torch.bmm(torch.bmm(Y_T, core), Y_T.transpose(1, 2))
    eye2  = torch.eye(2, device=device, dtype=dtype).expand(n, 2, 2)
    gamma_inv = torch.linalg.inv(gamma + gamma_reg * eye2)

    # ── Skorokhod integral δ ──────────────────────────────────────────────
    delta = torch.zeros(n, 2, device=device, dtype=dtype)

    for invYs, dW, g in zip(invY_list, dW_list, g_list):
        DsXT = g * torch.bmm(Y_T, invYs)                          # (n, 2, 2)
        U    = torch.bmm(DsXT.transpose(1, 2), gamma_inv)         # (n, 2, 2)
        delta += torch.bmm(U.transpose(1, 2), dW[:, :, None]).squeeze(-1)

    H = -delta
    return X_T, H


# ──────────────────────────────────────────────────────────────────────────────
# Reverse sampler — Euler–Maruyama
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def reverse_euler_nl(
    model,
    X_T: torch.Tensor,
    cfg: NonlinearSDEConfig,
    n_steps: int = 250,
    clamp_score: float = 50.0,
) -> torch.Tensor:
    """
    Euler–Maruyama reverse SDE (time T → 0):

        dX = [−b(X,t) + σ²β(t) s(X,t)] dt + σ√β(t) dW̄

    Parameters
    ----------
    model       : callable  (t_tensor (n,), x (n,2)) → score (n,2)
    X_T         : (n, 2)  starting positions at time T (from forward simulation)
    n_steps     : discretisation steps (should match forward dt = T/n_steps)
    clamp_score : hard clamp applied to predicted score

    Returns
    -------
    X_0 : (n, 2) on CPU
    """
    T       = cfg.T
    dt      = T / n_steps
    sqrt_dt = math.sqrt(dt)
    device  = X_T.device
    n       = X_T.shape[0]

    x = X_T.clone().to(device)

    for k in reversed(range(n_steps)):
        t_curr      = (k + 0.5) * dt
        beta        = beta_at(t_curr, cfg)
        g           = cfg.sigma * math.sqrt(beta)   # diffusion coefficient
        sigma2_beta = cfg.sigma**2 * beta           # g² = σ²β

        t_tensor = torch.full((n,), t_curr, device=device)
        score    = model(t_tensor, x)
        score    = torch.nan_to_num(score, nan=0.0).clamp(-clamp_score, clamp_score)

        # reverse drift: −b(X,t) + σ²β(t) · score
        fwd_drift = drift_nl(x, t_curr, cfg)
        rev_drift = -fwd_drift + sigma2_beta * score
        rev_drift = torch.nan_to_num(rev_drift, nan=0.0).clamp(-200.0, 200.0)

        x = x + rev_drift * dt + g * sqrt_dt * torch.randn_like(x)
        x = torch.nan_to_num(x, nan=0.0)

    return x.cpu()


@torch.no_grad()
def sample_stationary_nl(
    n_samples: int,
    cfg: NonlinearSDEConfig,
    dim: int = 2,
    device: str = "cuda",
    clamp: float = 20.0,
):
    """
    Sample approximate stationary distribution for Mirafzali nonlinear SDE.

    For the default setting k=1, sigma=1, a=0, each marginal is Cauchy-like:

        p_s(x) ∝ [1 + (x-a)^2]^(-k/sigma^2)

    For now we use the exact Cauchy sampler for the default k/sigma^2 = 1 case.
    """
    alpha = cfg.k / (cfg.sigma ** 2)

    if abs(alpha - 1.0) > 1e-8:
        print(
            f"[warning] sample_stationary_nl currently uses Cauchy approximation; "
            f"got k/sigma^2={alpha:.4f}"
        )

    loc = torch.tensor(float(cfg.a), device=device)
    scale = torch.tensor(1.0, device=device)

    dist = torch.distributions.Cauchy(loc=loc, scale=scale)
    x = dist.sample((n_samples, dim))

    if clamp is not None:
        x = x.clamp(-clamp, clamp)

    return x