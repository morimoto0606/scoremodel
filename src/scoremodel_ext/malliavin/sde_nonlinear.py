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


def hess_drift_nl(x: torch.Tensor, t: float, cfg: NonlinearSDEConfig) -> torch.Tensor:
    """
    Hessian tensor of the drift (second-order derivative).

    For the component-wise drift b_i(x) = -k β(t) (x_i-a) / (1+(x_i-a)²),
    the only nonzero entries are the fully-diagonal ones:

        H_{i,i,i}(x) = -k β(t) · 2(x_i-a)·((x_i-a)²-3) / (1+(x_i-a)²)³

    Parameters
    ----------
    x : (n, 2)

    Returns
    -------
    H : (n, 2, 2, 2)  —  H[:, i, j, k] = ∂²b_i / ∂x_j ∂x_k
    """
    beta  = beta_at(t, cfg)
    u     = x - cfg.a                         # (n, 2)
    u2    = u * u                              # (n, 2)
    denom = (1.0 + u2) ** 3                   # (n, 2)
    diag  = -cfg.k * beta * 2.0 * u * (u2 - 3.0) / denom   # (n, 2)

    n = x.shape[0]
    H = torch.zeros(n, 2, 2, 2, device=x.device, dtype=x.dtype)
    H[:, 0, 0, 0] = diag[:, 0]
    H[:, 1, 1, 1] = diag[:, 1]
    return H


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

def simulate_malliavin_nl(
    X0: torch.Tensor,
    T: float,
    cfg: NonlinearSDEConfig,
    n_steps: int = 250,
    gamma_reg: float = 1e-3,
    correction: str = "approx",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Euler–Maruyama forward simulation with Malliavin score weights.

    Dispatches to a specific implementation via *correction*:

    ``"approx"`` (default)
        First-variation-only (Y only, no Z).  Fast and numerically stable.
        Equivalent to the Bismut–Elworthy–Li formula for the nonlinear SDE.

    ``"a_correction"`` (experimental)
        Tracks the second variation Z and applies the leading A-correction to
        the Skorokhod divergence (Mirafzali Algorithm 4+5, partial).
        B/C double-integral corrections are *not* included — see
        ``_simulate_malliavin_nl_a_correction`` docstring.

    Parameters
    ----------
    X0         : (n, 2)
    T          : terminal time
    n_steps    : Euler steps
    gamma_reg  : Tikhonov regularisation for γ inversion
    correction : one of ``"approx"`` | ``"a_correction"``

    Returns
    -------
    X_T : (n, 2) terminal positions
    H   : (n, 2) Malliavin score weights s.t. ∇_x log p_T ≈ E[H | X_T = x]
    """
    if correction == "approx":
        return _simulate_malliavin_nl_approx(
            X0, T, cfg, n_steps=n_steps, gamma_reg=gamma_reg
        )
    if correction == "a_correction":
        return _simulate_malliavin_nl_a_correction(
            X0, T, cfg, n_steps=n_steps, gamma_reg=gamma_reg
        )
    raise ValueError(
        f"Unknown correction {correction!r}. "
        "Choose from 'approx' or 'a_correction'."
    )


@torch.no_grad()
def _simulate_malliavin_nl_a_correction(
    X0: torch.Tensor,
    T: float,
    cfg: NonlinearSDEConfig,
    n_steps: int = 250,
    gamma_reg: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    **Experimental** — Mirafzali Algorithms 4+5 with the A-correction term.

    Tracks X, Y (first variation), Z (second variation) in the forward pass,
    then computes the Malliavin covariance γ and the Skorokhod integral
    S − D where D is the leading A-correction from the derivative of γ⁻¹.

    .. note::
        The B and C double-integral corrections from Appendix D.4 of Mirafzali
        et al. are **not** implemented here (they require an O(N²) nested loop
        over time steps).  This function therefore implements only a partial
        Algorithm 5.  Use ``correction='approx'`` for the stable default.

    Z update (Algorithm 4, using old-step values):
        J, H evaluated at (x_old, t)
        T1 = einsum('bimp,bmj,bpl->bijl', H, Y_old, Y_old)
        T2 = einsum('bim,bmjl->bijl',     J, Z_old)
        Y_new = Y_old + J @ Y_old * dt
        Z_new = Z_old + (T1 + T2) * dt
        x_new = x_old + b * dt + g * dW

    A-correction (single deterministic integral):
        term1_{b,k} = g² Σ_{i,j,l} (Y_t⁻¹)_{b,i,j} Z_T_{b,i,j,l} (γ⁻¹)_{b,l,k}
        term2_{b,k} = g² Σ_{i,j,l,m} (Y_t⁻¹)_{b,i,j} Z_t_{b,i,j,l}
                          (Y_t⁻¹)_{b,l,m} Y_T_{b,p,m} (γ⁻¹)_{b,p,k}
        D = Σ_t (term1 − term2) * dt
        δ = S − D,    H = −δ

    Parameters
    ----------
    X0        : (n, 2)
    T         : terminal time
    n_steps   : Euler steps
    gamma_reg : Tikhonov regularisation for γ inversion

    Returns
    -------
    X_T : (n, 2) terminal positions
    H   : (n, 2) Malliavin score weights
    """
    n       = X0.shape[0]
    device  = X0.device
    dtype   = X0.dtype
    dt      = T / n_steps
    sqrt_dt = math.sqrt(dt)

    x = X0.clone()
    Y = torch.eye(2, device=device, dtype=dtype).expand(n, 2, 2).clone()
    Z = torch.zeros(n, 2, 2, 2, device=device, dtype=dtype)

    Y_all:  list = []
    Z_all:  list = []
    dW_all: list = []
    g_all:  list = []

    # ── Algorithm 4: forward pass (use old-step values for Y/Z updates) ──
    for k in range(n_steps):
        t_mid  = (k + 0.5) * dt
        g      = sigma_t(t_mid, cfg)

        # Save old values *before* the Euler step
        Y_old = Y
        Z_old = Z
        Y_all.append(Y_old.clone())
        Z_all.append(Z_old.clone())
        g_all.append(g)

        dW = sqrt_dt * torch.randn(n, 2, device=device, dtype=dtype)
        dW_all.append(dW)

        # Evaluate J and H at current (old) x
        J      = jac_drift_nl(x, t_mid, cfg)    # (n,2,2)
        H_hess = hess_drift_nl(x, t_mid, cfg)   # (n,2,2,2)

        # Euler updates using old Y, Z
        # T1[b,i,j,l] = Σ_{m,p} H[b,i,m,p] Y_old[b,m,j] Y_old[b,p,l]
        T1 = torch.einsum('bimp,bmj,bpl->bijl', H_hess, Y_old, Y_old)
        # T2[b,i,j,l] = Σ_m J[b,i,m] Z_old[b,m,j,l]
        T2 = torch.einsum('bim,bmjl->bijl', J, Z_old)

        x = x + drift_nl(x, t_mid, cfg) * dt + g * dW
        Y = Y_old + torch.bmm(J, Y_old) * dt
        Z = Z_old + (T1 + T2) * dt

    X_T = x
    Y_T = Y
    Z_T = Z

    # ── Malliavin covariance γ ────────────────────────────────────────────
    gamma    = torch.zeros(n, 2, 2, device=device, dtype=dtype)
    invY_all: list = []
    W_all:    list = []

    for Ys, gs in zip(Y_all, g_all):
        invYs = torch.linalg.inv(Ys)
        Ws    = gs * torch.bmm(Y_T, invYs)    # W_s = g_s Y_T Y_s⁻¹
        gamma += torch.bmm(Ws, Ws.transpose(1, 2)) * dt
        invY_all.append(invYs)
        W_all.append(Ws)

    eye2      = torch.eye(2, device=device, dtype=dtype).expand(n, 2, 2)
    gamma_inv = torch.linalg.inv(gamma + gamma_reg * eye2)

    # ── Stochastic term S ─────────────────────────────────────────────────
    S = torch.zeros(n, 2, device=device, dtype=dtype)
    for Ws, dW in zip(W_all, dW_all):
        ginvW = torch.bmm(gamma_inv, Ws)
        S    += torch.bmm(ginvW.transpose(1, 2), dW.unsqueeze(-1)).squeeze(-1)

    # ── A-correction D (single deterministic integral) ────────────────────
    # TODO: B/C double-integral corrections (Appendix D.4) not yet implemented.
    D = torch.zeros(n, 2, device=device, dtype=dtype)

    for invYt, Zt, gt in zip(invY_all, Z_all, g_all):
        g2 = gt * gt
        # term1: g² Σ_{i,j,l} (Y_t⁻¹)_{b,i,j} Z_T_{b,i,j,l} (γ⁻¹)_{b,l,k}
        t1 = g2 * torch.einsum('bij,bijl,blk->bk', invYt, Z_T, gamma_inv) * dt

        # term2: g² Σ_{i,j,l,m,p} (Y_t⁻¹)_{b,i,j} Z_t_{b,i,j,l}
        #            (Y_t⁻¹)_{b,l,m} Y_T_{b,p,m} (γ⁻¹)_{b,p,k}
        c  = torch.einsum('bij,bijl->bl', invYt, Zt)        # (n,2)
        c2 = torch.einsum('bl,bml->bm', c, invYt)           # (n,2)
        t2 = g2 * torch.einsum('bm,bpm,bpk->bk', c2, Y_T, gamma_inv) * dt

        D += t1 - t2

    H = -(S - D)
    return X_T, H

@torch.no_grad()
def _simulate_malliavin_nl_approx(
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