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
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

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


def hess_drift_diag_nl(x: torch.Tensor, t: float, cfg: NonlinearSDEConfig) -> torch.Tensor:
    """
    Diagonal second derivative of the drift (Hessian):

        ∂²b_i/∂x_i² = −k β(t) · 2(x_i − a)((x_i − a)² − 3) / (1 + (x_i − a)²)³

    Off-diagonal entries are zero.

    Parameters
    ----------
    x : (n, 2)

    Returns
    -------
    hess : (n, 2)  — only the diagonal d²b_i/dx_i² for i in {0,1}
    """
    beta = beta_at(t, cfg)
    u    = x - cfg.a                                     # (n, 2)
    u2   = u * u
    num  = 2.0 * u * (u2 - 3.0)
    denom = (1.0 + u2) ** 3
    return -cfg.k * beta * num / denom                   # (n, 2)


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
    return_diagnostics: bool = False,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, Dict[str, float]],
]:
    """
    Dispatch to the requested nonlinear Malliavin-weight implementation.

    Parameters
    ----------
    correction : "approx" (default), "a_correction", or "mirafzali_full"
        "approx" is the stable first-variation-only implementation.
        "a_correction" adds the experimental A-correction only.
        "mirafzali_full" currently calls a diagonal specialized attempt,
        not a verified faithful implementation of the paper's full Algorithm 5.
    """
    if correction == "approx":
        return simulate_malliavin_nl_approx(
            X0,
            T,
            cfg,
            n_steps=n_steps,
            gamma_reg=gamma_reg,
            return_diagnostics=return_diagnostics,
        )
    if correction == "a_correction":
        out = _simulate_malliavin_nl_a_correction(
            X0, T, cfg, n_steps=n_steps, gamma_reg=gamma_reg
        )
        if not return_diagnostics:
            return out
        X_T, H = out
        h_norm = torch.linalg.norm(H, dim=1)
        diagnostics = {
            "var_H": float(H.var(unbiased=False).item()),
            "mean_H_norm": float(h_norm.mean().item()),
            "var_H_norm": float(h_norm.var(unbiased=False).item()),
        }
        return X_T, H, diagnostics
    if correction == "mirafzali_full":
        return simulate_malliavin_nl_mirafzali_full(
            X0,
            T,
            cfg,
            n_steps=n_steps,
            gamma_reg=gamma_reg,
            return_diagnostics=return_diagnostics,
        )
    raise ValueError(
        f"Unknown correction={correction!r}; expected 'approx', 'a_correction', or 'mirafzali_full'"
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
def simulate_malliavin_nl_approx(
    X0: torch.Tensor,
    T: float,
    cfg: NonlinearSDEConfig,
    n_steps: int = 250,
    gamma_reg: float = 1e-3,
    return_diagnostics: bool = False,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, Dict[str, float]],
]:
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
    if not return_diagnostics:
        return X_T, H

    h_norm = torch.linalg.norm(H, dim=1)
    diagnostics = {
        "var_H": float(H.var(unbiased=False).item()),
        "mean_H_norm": float(h_norm.mean().item()),
        "var_H_norm": float(h_norm.var(unbiased=False).item()),
    }
    return X_T, H, diagnostics


# Backwards compatibility: older tests and experiments imported the leading-
# underscore name directly.
_simulate_malliavin_nl_approx = simulate_malliavin_nl_approx


# ──────────────────────────────────────────────────────────────────────────────
# Full Algorithm 4 + 5 implementation (Mirafzali et al.)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def simulate_malliavin_nl_mirafzali_full(
    X0: torch.Tensor,
    T: float,
    cfg: NonlinearSDEConfig,
    n_steps: int = 250,
    gamma_reg: float = 1e-3,
    return_diagnostics: bool = False,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, Dict[str, float]],
]:
    """
    Diagonal specialized implementation attempt for Algorithm 4 + 5.

    This implementation uses the exact diagonal specialization of Algorithm 5
    for the component-wise nonlinear SDE in Appendix C.  Since the drift
    Jacobian, Hessian, flow Y, second variation Z, Malliavin covariance, W,
    Omega, I1, I2, A, B and C are all diagonal for this model, the matrix/tensor
    equations reduce to element-wise scalar equations for each coordinate.
    The B and C terms are still computed through the Algorithm 5 time
    integrals; they are not replaced by the previous I1(u,u)-only shortcut.

    Tracks the second variation Z (3rd-order tensor) in addition to X and Y,
    then computes correction terms Ω, Θ, I₁, I₂, A, B, C and the
    deterministic correction D so that the Skorokhod integral is

        δ_{t_k}(u_{t_k}) = S − D
        H = −δ

    where (Algorithm 5, lines 24–26):
        S = γ⁻¹_{X_{t_k}} Y_{t_k} ∫₀^{t_k} (Y_u⁻¹ σ(u))ᵀ dB_u
        D = ∫₀^{t_k} (Y_u⁻¹ σ(u))ᵀ [A − B − C](u, t_k) du

    and (Algorithm 5, lines 20–22):
        A(u,t_k) = [σ(u)ᵀ (Y_u⁻¹)ᵀ (Z_{t_k} − Z_u (Y_u⁻¹)ᵀ Y_{t_k})] γ⁻¹_{X_{t_k}}
        B(u,t_k) = Y_{t_k}ᵀ γ⁻¹_{X_{t_k}} [∫₀ᵘ I₁(u,v) dv] γ⁻¹_{X_{t_k}}
        C(u,t_k) = Y_{t_k}ᵀ γ⁻¹_{X_{t_k}} [∫₀^{t_k} I₂(u,v) dv] γ⁻¹_{X_{t_k}}

    Parameters
    ----------
    X0        : (n, 2)
    T         : terminal time
    n_steps   : Euler steps
    gamma_reg : Tikhonov regularisation for γ inversion

    Returns
    -------
    X_T : (n, 2)
    H   : (n, 2)
    """
    n      = X0.shape[0]
    d      = 2
    device = X0.device
    dtype  = X0.dtype
    dt     = T / n_steps
    sqrt_dt = math.sqrt(dt)

    x = X0.clone()
    Y = torch.eye(d, device=device, dtype=dtype).expand(n, d, d).clone()
    # Z[i, a, b] = ∂²X_i / ∂x_a⁰ ∂x_b⁰ ; shape (n, d, d, d)
    Z = torch.zeros(n, d, d, d, device=device, dtype=dtype)

    X_list:  list = []
    Y_list:  list = []
    Z_list:  list = []
    dW_list: list = []
    g_list:  list = []

    # ── Algorithm 4 forward pass ───────────────────────────────────────────
    for k in range(n_steps):
        t_mid = (k + 0.5) * dt
        g     = sigma_t(t_mid, cfg)

        X_list.append(x.clone())
        Y_list.append(Y.clone())
        Z_list.append(Z.clone())
        g_list.append(g)

        dW = sqrt_dt * torch.randn(n, d, device=device, dtype=dtype)
        dW_list.append(dW)

        # Euler step for X
        x_new = x + drift_nl(x, t_mid, cfg) * dt + g * dW

        # Euler step for Y: dY = J Y dt
        J = jac_drift_nl(x, t_mid, cfg)                     # (n, d, d)
        Y_new = Y + torch.bmm(J, Y) * dt

        # Euler step for Z: dZ_{iab} = Σ_j h_j δ_{ij} Y_{ja} Y_{jb} dt + Σ_j J_{ij} Z_{jab} dt
        # For diagonal J and diagonal hessian:
        #   dZ_{iab} = h_i Y_{ia} Y_{ib} dt + J_{ii} Z_{iab} dt
        h = hess_drift_diag_nl(x, t_mid, cfg)               # (n, d)  diagonal hessian
        # h_i Y_{ia} Y_{ib}: einsum 'ni,nia,nib->niab'
        hYY = h[:, :, None, None] * Y[:, :, :, None] * Y[:, :, None, :]   # (n, d, d, d)
        # J_{ii} Z_{iab}: einsum 'nij,njab->niab' but J is diagonal so just diag*Z
        JZ  = J.diagonal(dim1=1, dim2=2)[:, :, None, None] * Z             # (n, d, d, d)
        Z_new = Z + (hYY + JZ) * dt

        x, Y, Z = x_new, Y_new, Z_new

    X_T = x
    Y_T = Y        # (n, d, d)
    Z_T = Z        # (n, d, d, d)

    # ── Malliavin covariance γ (same as approx) ────────────────────────────
    inv_Y_list: list = []
    core = torch.zeros(n, d, d, device=device, dtype=dtype)
    for Ys, g in zip(Y_list, g_list):
        invYs = torch.linalg.inv(Ys)                        # (n, d, d)
        inv_Y_list.append(invYs)
        core += g**2 * torch.bmm(invYs, invYs.transpose(1, 2)) * dt

    gamma     = torch.bmm(torch.bmm(Y_T, core), Y_T.transpose(1, 2))   # (n,d,d)
    eye2      = torch.eye(d, device=device, dtype=dtype).expand(n, d, d)
    gamma_inv = torch.linalg.inv(gamma + gamma_reg * eye2)              # (n,d,d)

    # ── Algorithm 5 correction terms ──────────────────────────────────────
    # Pre-compute W_s = g(s) Y_T inv(Y_s), shape (n,d,d)
    W_list: list = []
    for invYs, g in zip(inv_Y_list, g_list):
        Ws = g * torch.bmm(Y_T, invYs)   # (n,d,d)
        W_list.append(Ws)

    # Pre-compute Ω(s) for each step s:
    # Ω(s) = g(s) Z_T einv_s  −  g(s) Y_T einv_s Z_s einv_s
    # where einv_s = inv(Y_s), shapes chosen so Ω(s) has shape (n, d, d, d)
    # i.e. Ω_iab = g [ Z_T_{iaj} (inv_Ys)_{jb} − (Y_T inv_Ys)_{ij} Z_s_{jab} (inv_Ys ... wait
    # From paper: Ω(t) is built from second-variation; let us keep (n,d,d,d) as:
    # Omega[n, i, a, b] = Z_T[n,i,j,a] (inv_Ys)[n,j,b]  (contraction on j)
    #                   − (Y_T inv_Ys)[n,i,j] Z_s[n,j,a,b]  (contraction on j)
    # but g factor folds into W, so for I_1 and I_2 we need:
    # I_1(s,v) = [Ω(s)(Y_s^{-1})^{-1}σ(s)] W_v^T + W_v [Ω(s)(Y_v^{-1})^{-1}σ(s)]^T
    # This is complex; let us use a simpler factored form.
    #
    # Ω(t) contracted with g(t)(Y_t^{-1})^T gives a (n,d,d) matrix M_t:
    # M_t = g(t) [ Z_T @ inv(Y_t) - Y_T @ inv(Y_t) @ Z_t @ inv(Y_t) ] @ inv(Y_t)^T @ σ
    # For scalar σ=g this simplifies. We define:
    # Phi_s = g(s)^2 [Z_T einv_s einv_s^T - Y_T einv_s Z_s einv_s einv_s^T]  (n,d,d,d)

    # For I_1, I_2 we need (n,d,d) matrices at each pair (s,v).
    # Let's define M(s) = g(s) Omega_contracted_s  shape (n,d,d,d) → contracted to (n,d,d)
    # Actually from the paper lines 17-18:
    # I_1(t,s) = [Omega(t)(Y_s^{(i)})^{-1} sigma(s)] W_s^T + W_s [...]^T
    # The argument (Y_s^{(i)})^{-1} sigma(s) is g(s)*inv(Y_s) = (1/g)*W_s^T ... 
    # Simplified: let P_s = g(s) einsum 'niab,nbj->niaj'(Z_T_partial, invYs) - Y_T einv_s Z_s_partial einv_s
    # This gets complicated with arbitrary Omega definition. We implement directly:

    # For each step u, we need A(u), B(u), C(u), each (n,d,d):
    # A(u,t_k) = [σ(u) (Y_u^{-1})^T (Z_T - Z_u (Y_u^{-1})^T Y_T)] γ^{-1}
    #   Note: σ(u)=g(u)*I so g(u)*(inv_Yu)^T @ (Z_T - Z_u @ inv_Yu^T @ Y_T) contracts to (n,d,d)
    #   Z_T has shape (n,d,d,d), we need: for each (a,b) index of the output,
    #   A[n,a,b] = g(u) Σ_i (inv_Yu^T)[n,a,i] (Z_T[n,i,:,b] - Z_u[n,i,:,:] @ (inv_Yu^T @ Y_T)[n,:,b])
    #   Then multiply by gamma_inv: result (n,d,d) @ (n,d,d) -> (n,d,d)
    # B(u,t_k) = Y_T^T gamma_inv [∫₀^u I_1(u,v) dv] gamma_inv  -> (n,d,d)
    # C(u,t_k) = Y_T^T gamma_inv [∫₀^{t_k} I_2(u,v) dv] gamma_inv -> (n,d,d)

    # For I_1(t,s): using Algorithm 5, line 17 with Omega(t)(Y_s)^{-1} sigma(s):
    # Omega(t) as a map (n,d,d,d): Omega_iab = g(t) [Z_T_{iac} inv_Yt_{cb} - Y_T_{ij} inv_Yt_{jc} Z_t_{cab} inv_... ]
    # This is messy. We implement Omega as (n,d,d,d) using einsum notation:

    # Omega[n,i,a,b] = g(t) * (
    #     sum_c Z_T[n,i,a,c] * inv_Yt[n,c,b]
    #   - sum_j Y_T[n,i,j] * sum_c inv_Yt[n,j,c] * Z_t[n,c,a,b]  ← NOTE: this would be (n,d,d,d)
    # )
    # Then [Omega(t) (Y_s^{-1}) sigma(s)][n,i,a] = g(s) * sum_b Omega[n,i,a,b] * inv_Ys[n,b,?]
    # But sigma(s)=g(s)*I, so sigma acts on last index: sum_b Omega[n,i,a,b]*g(s)*delta_{b,?}
    # => Omega_contracted[n,i,a] = g(s) * Omega[n,i,a,:] summed appropriately...
    # This is getting complex. We'll implement step-by-step below.

    # Precompute for all steps: invYs, Ys, Zs, Ws
    # Then loop u over all steps for S stochastic integral
    # For A, B, C: loop u, and for B/C also need inner integral over v

    YT_t = Y_T.transpose(1, 2)    # (n,d,d)  = Y_T^T

    # Precompute running integrals needed for B and C
    # int_I1_cumul[u] = ∫₀ᵘ I_1(u,v) dv  (but I_1 depends on u, so we can't precompute naively)
    # Instead: for each u, compute integral over v=0..u of I_1(u,v)*dt
    # I_1(t,s) = Omega_vec(t,s) W_s^T + W_s Omega_vec(t,s)^T  where Omega_vec shape (n,d,d)
    # Omega_vec(t,s) = Omega(t) @ inv(Y_s) @ sigma(s) = [Omega(t) applied to g(s)*inv_Ys cols]
    # Omega(t)[n,i,a,b] = g(t)*[Z_T[n,i,a,?]inv_Yt - Y_T inv_Yt Z_t[n,:,a,?]inv_Yt][?,b]
    # Let's define cleanly:
    # For each step t:
    #   F_t[n,i,a,b] = sum_c Z_T[n,i,a,c] * inv_Yt[n,c,b]   (n,d,d,d)
    #   G_t[n,i,a,b] = sum_j (Y_T inv_Yt)[n,i,j] * Z_t[n,j,a,b]  (n,d,d,d)
    #   Omega_t = g_t * (F_t - G_t)   (n,d,d,d)
    # Then Omega_vec(t,s)[n,i,a] = g_s * sum_b Omega_t[n,i,a,b] * inv_Ys[n,b,0]... 
    # Wait: sigma(s) = g_s * I_{d×d}, so:
    #   [Omega(t) * sigma(s)][n,i,a,c] = g_s * Omega_t[n,i,a,c]  (just scalar multiply)
    # Then contract with inv_Ys: sum_c [Omega(t)*sigma(s)][n,i,a,c] * inv_Ys[n,c,j]
    #   => g_s * sum_c Omega_t[n,i,a,c] * inv_Ys[n,c,j]  shape (n,d,d,d)
    # But I_1(t,s) is (n,d,d), so we need to reduce the last (a,?) dims somehow.
    # Looking at line 17 more carefully: I_1(t,s) = [Ω(t)(Y_s^{(i)})^{-1}σ(s)] W_s^T + W_s [...]^T
    # The bracket [Ω(t)(Y_s^{(i)})^{-1}σ(s)] must be shape (n,d,d) for I_1 to be (n,d,d).
    # So Ω(t) applied to (Y_s^{-1}σ(s)) — where (Y_s^{-1}σ(s)) is (n,d,d) — yields (n,d,d).
    # Ω(t) must be a (n,d,d)→(n,d,d) linear map, i.e., (n,d,d,d) tensor contraction on last index:
    # result[n,i,a] = sum_b Omega[n,i,a,b] * [inv_Ys sigma(s)][n,b,?]  — this gives (n,d,d) if we fix one output dim
    # Actually: [Omega(t)(Y_s^{-1}sigma(s))][n,i,?] = sum_b sum_j Omega[n,i,b] (n,d,d treated as matrix) * (inv_Ys*g_s)[n,b,?]
    # => Omega must be (n,d,d) → (n,d,d). So Omega is a (n,d,d) matrix times g_t... Let me reconsider.
    # 
    # From the paper (supplementary): Omega(t) = dW_t/dt (Malliavan derivative of W_t)
    # W_t = g(t) Y_T Y_t^{-1}, so dW_t/dX_0 involves Z.
    # Ω(t) as a (n,d,d) matrix (not 4th order): 
    # Omega(t)[n,i,a] = g(t) [Z_T[n,i,:,a] - Y_T[n,i,:] @ inv_Yt[n,:,:] @ Z_t[n,:,:,a]] @ inv_Yt[n,:,?]
    # This needs more careful reading. Given the complexity, we use a pragmatic dimension analysis:
    # W_s shape (n,d,d). I_1(t,s) shape (n,d,d). So [Omega(t)(Y_s^{-1}sigma(s))] is (n,d,d).
    # Let Omega(t) be a (n,d,d) matrix = dW_t/dt * dt ... 
    # Simplest consistent definition: Omega_t = (n,d,d) where
    # Omega_t = g_t * [Z_T_mat @ inv_Yt - Y_T @ inv_Yt @ Z_t_mat @ inv_Yt]
    # where Z_mat collapses one index: Z_mat[n,i,b] = sum_a Z[n,i,a,b] ? No, Z is 3-index...
    # 
    # Exact diagonal specialization.
    #
    # For the Appendix C SDE, each coordinate evolves independently and the
    # drift is component-wise.  Therefore J, H, Y, Z, gamma, W, Omega, I1, I2,
    # A, B and C remain diagonal if initialized diagonally.  We may therefore
    # work coordinate-by-coordinate with scalar diagonal entries:
    #
    #   y_i(t)      = Y_t[i,i]
    #   z_i(t)      = Z_t[i,i,i]
    #   gamma_i     = gamma[i,i]
    #   w_i(v)      = g(v) y_i(T) / y_i(v)
    #   omega_i(u)  = g(u)/y_i(u) * (z_i(T) - y_i(T) z_i(u)/y_i(u))
    #
    # The Algorithm 5 integrals reduce to
    #
    #   I1_i(u,v) = 2 omega_i(u) * g(v)/y_i(v) * w_i(v),
    #   I2_i(u,v) = 2 omega_i(u) * g(v)/y_i(v) * w_i(v),
    #
    # with different integration ranges:
    #
    #   ∫_0^u I1_i(u,v) dv,
    #   ∫_0^T I2_i(u,v) dv.
    #
    # This is no longer the old I1(u,u)-only shortcut.

    # Recompute z_diag for all stored Z_list
    # Z_list stores full (n,d,d,d) tensors from the forward pass; extract diagonal
    z_diag_list = []
    for Zs in Z_list:
        # Z[n,i,i,i] for i in {0,1}
        z_d = torch.stack([Zs[:, i, i, i] for i in range(d)], dim=1)   # (n,d)
        z_diag_list.append(z_d)
    z_diag_T = torch.stack([Z_T[:, i, i, i] for i in range(d)], dim=1)  # (n,d)

    # Similarly Y and inv_Y are diagonal for diagonal J (starting from I):
    # Y[n,i,j]=0 for i≠j, Y[n,i,i] = product of exp(int J_{ii} dt)
    # Store y_diag = Y diagonals, inv_y_diag = inv_Y diagonals
    y_diag_list   = [torch.stack([Ys[:, i, i] for i in range(d)], dim=1) for Ys in Y_list]
    inv_y_diag_list = [torch.stack([invYs[:, i, i] for i in range(d)], dim=1) for invYs in inv_Y_list]
    y_diag_T      = torch.stack([Y_T[:, i, i] for i in range(d)], dim=1)   # (n,d)
    inv_y_diag_T  = torch.linalg.inv(Y_T).diagonal(dim1=1, dim2=2)         # (n,d)

    # Under diagonal structure, gamma is also diagonal:
    # gamma[n,i,i] = Y_T[n,i,i]^2 * sum_s inv_Ys[n,i,i]^2 * g_s^2 * dt
    # gamma_inv[n,i,i] = 1 / (gamma[n,i,i] + gamma_reg)
    gamma_inv_diag = gamma_inv.diagonal(dim1=1, dim2=2)  # (n,d)

    # W_s[n,i,i] = g_s * y_diag_T[n,i] * inv_y_diag_s[n,i]
    w_diag_list = [g * y_diag_T * inv_y_diag_s for g, inv_y_diag_s in zip(g_list, inv_y_diag_list)]
    # (n,d) each

    # Omega_t[n,i] (diagonal, shape (n,d)):
    # Omega_t[n,i] = g_t * (z_diag_T[n,i] * inv_y_diag_t[n,i]
    #                      - y_diag_T[n,i] * inv_y_diag_t[n,i] * z_diag_t[n,i] * inv_y_diag_t[n,i])
    # = g_t * inv_y_diag_t[n,i] * (z_diag_T[n,i] - y_diag_T[n,i] * inv_y_diag_t[n,i] * z_diag_t[n,i])
    omega_diag_list = []
    for g, inv_y_diag_s, z_diag_s in zip(g_list, inv_y_diag_list, z_diag_list):
        # omega_i(u) = g(u) / y_i(u) * (z_i(T) - y_i(T) z_i(u) / y_i(u))
        omega = g * inv_y_diag_s * (z_diag_T - y_diag_T * inv_y_diag_s * z_diag_s)
        omega_diag_list.append(omega)

    # For diagonal matrices, the Algorithm 5 matrix products reduce to
    # element-wise products.  Define
    #     q_i(v) = g(v) / y_i(v) * w_i(v)
    #            = g(v)^2 y_i(T) / y_i(v)^2.
    # Then both diagonal integrands reduce to
    #     I1_i(u,v) = 2 omega_i(u) q_i(v),
    #     I2_i(u,v) = 2 omega_i(u) q_i(v),
    # with I1 integrated over v <= u and I2 over all v <= T.

    # ── Stochastic term S ─────────────────────────────────────────────────
    # S[n,i] = gamma_inv[n,i] * y_diag_T[n,i] * sum_u (g_u * inv_y_diag_u[n,i] * dW_u[n,i])
    stoch_sum = torch.zeros(n, d, device=device, dtype=dtype)
    for inv_y_d, g, dW in zip(inv_y_diag_list, g_list, dW_list):
        stoch_sum += g * inv_y_d * dW   # (n,d)

    S = gamma_inv_diag * y_diag_T * stoch_sum   # (n,d)

    # ── Correction terms A, B, C and deterministic term D ─────────────────
    # D[n,i] = sum_u g_u * inv_y_diag_u[n,i] * (A-B-C)(u, t_k)[n,i] * dt
    # A(u,t_k)[n,i] = g_u * inv_y_diag_u[n,i] * (z_diag_T[n,i] - z_diag_u[n,i]*inv_y_diag_u[n,i]*y_diag_T[n,i]) * gamma_inv_diag[n,i]
    # B(u,t_k)[n,i] = y_diag_T[n,i] * gamma_inv_diag[n,i] * cumI1_u[n,i] * gamma_inv_diag[n,i]
    # C(u,t_k)[n,i] = y_diag_T[n,i] * gamma_inv_diag[n,i] * totalI2_u[n,i] * gamma_inv_diag[n,i]

    # q_i(v) = g(v) / y_i(v) * w_i(v) = g(v)^2 y_i(T) / y_i(v)^2.
    q_list = [g * inv_y_d * w_d for g, inv_y_d, w_d in zip(g_list, inv_y_diag_list, w_diag_list)]

    # Prefix integrals for ∫_0^u q(v) dv and total integral for ∫_0^T q(v) dv.
    q_prefix_list = []
    q_prefix = torch.zeros(n, d, device=device, dtype=dtype)
    for q in q_list:
        q_prefix = q_prefix + q * dt
        q_prefix_list.append(q_prefix.clone())
    q_total = q_prefix

    D = torch.zeros(n, d, device=device, dtype=dtype)
    mean_A2_sum = 0.0
    mean_B2_sum = 0.0
    mean_C2_sum = 0.0
    n_abc = 0

    for inv_y_d, g, z_d, omega_d, q_prefix_u in zip(
        inv_y_diag_list, g_list, z_diag_list, omega_diag_list, q_prefix_list
    ):
        # A_i(u) = g(u)/y_i(u) * (z_i(T) - y_i(T) z_i(u)/y_i(u)) * gamma_i^{-1}
        A_u = g * inv_y_d * (z_diag_T - y_diag_T * inv_y_d * z_d) * gamma_inv_diag

        # B_i(u) = y_i(T) gamma_i^{-1} [∫_0^u I1_i(u,v) dv] gamma_i^{-1}
        int_I1_u = 2.0 * omega_d * q_prefix_u
        B_u = y_diag_T * gamma_inv_diag * int_I1_u * gamma_inv_diag

        # C_i(u) = y_i(T) gamma_i^{-1} [∫_0^T I2_i(u,v) dv] gamma_i^{-1}
        int_I2_u = 2.0 * omega_d * q_total
        C_u = y_diag_T * gamma_inv_diag * int_I2_u * gamma_inv_diag

        mean_A2_sum += float((A_u * A_u).mean().item())
        mean_B2_sum += float((B_u * B_u).mean().item())
        mean_C2_sum += float((C_u * C_u).mean().item())
        n_abc += 1

        # D_i = ∫ g(u)/y_i(u) [A_i(u) - B_i(u) - C_i(u)] du.
        D += g * inv_y_d * (A_u - B_u - C_u) * dt

    # ── Assemble δ and H ─────────────────────────────────────────────────
    delta = S - D
    H     = -delta
    if not return_diagnostics:
        return X_T, H

    h_norm = torch.linalg.norm(H, dim=1)
    denom = max(n_abc, 1)
    diagnostics = {
        "var_S": float(S.var(unbiased=False).item()),
        "var_D": float(D.var(unbiased=False).item()),
        "var_delta": float(delta.var(unbiased=False).item()),
        "var_H": float(H.var(unbiased=False).item()),
        "mean_H_norm": float(h_norm.mean().item()),
        "var_H_norm": float(h_norm.var(unbiased=False).item()),
        "mean_A2": float(mean_A2_sum / denom),
        "mean_B2": float(mean_B2_sum / denom),
        "mean_C2": float(mean_C2_sum / denom),
    }
    return X_T, H, diagnostics




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