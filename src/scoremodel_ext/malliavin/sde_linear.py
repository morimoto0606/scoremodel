"""
Linear SDE schedules (VE / VP / sub-VP) with direct (reparameterised)
forward simulation and closed-form Malliavin score weights.

Malliavin weight identity
-------------------------
For a linear SDE  dX = f(X, t) dt + g(t) dW  (additive or linear drift),
the Malliavin score weight per path satisfies

    ∇_x log p_T(x) ≈ E[ H(ω) | X_T(ω) = x ]

where H is computed analytically below for each SDE type.

VE  (Variance Exploding)
    dX = σ(t) dW,   σ(t) = σ_min · (σ_max/σ_min)^{t/T}
    X_T | X_0 ~ N(X_0, Σ²(T) I),  Σ²(T) = ∫_0^T σ²(t) dt
    H = (X_0 − X_T) / Σ²(T)

VP  (Variance Preserving)
    dX = −½ β(t) X dt + √β(t) dW,   β(t) = β_min + (β_max−β_min)·t/T
    X_T | X_0 ~ N(α_T X_0, (1−α_T²) I),  α_T = exp(−B(T)/2)
    H = (α_T X_0 − X_T) / (1 − α_T²)

sub-VP
    Same drift as VP; diffusion coefficient √(β(t)(1−e^{−B(t)}))
    X_T | X_0 ~ N(α_T X_0, v_T I),
    v_T = (1 − α_T²) − B(T) α_T²   (see derivation in docstring)
    H = (α_T X_0 − X_T) / v_T
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

import torch


# ──────────────────────────────────────────────────────────────────────────────
# SDE configuration dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class VEConfig:
    """Variance Exploding SDE configuration."""
    sigma_min: float = 0.1
    sigma_max: float = 10.0
    T: float = 1.0


@dataclass
class VPConfig:
    """Variance Preserving SDE configuration."""
    beta_min: float = 0.1
    beta_max: float = 20.0
    T: float = 1.0


@dataclass
class SubVPConfig:
    """Sub-Variance Preserving SDE configuration."""
    beta_min: float = 0.1
    beta_max: float = 20.0
    T: float = 1.0


# ──────────────────────────────────────────────────────────────────────────────
# VE helpers
# ──────────────────────────────────────────────────────────────────────────────

def ve_sigma_at(t: float, cfg: VEConfig) -> float:
    """σ(t) for the VE schedule."""
    if abs(cfg.sigma_max - cfg.sigma_min) < 1e-8:
        return cfg.sigma_min
    return cfg.sigma_min * (cfg.sigma_max / cfg.sigma_min) ** (t / cfg.T)


def ve_marginal_var(T: float, cfg: VEConfig) -> float:
    """
    Σ²(T) = ∫_0^T σ²(t) dt

    For the geometric schedule σ(t) = σ_min · (σ_max/σ_min)^{t/T}:
      Σ²(T) = T · (σ_max² − σ_min²) / (2 · ln(σ_max/σ_min))

    For constant σ (σ_min == σ_max):
      Σ²(T) = σ² · T
    """
    if abs(cfg.sigma_max - cfg.sigma_min) < 1e-8:
        return cfg.sigma_min ** 2 * T
    log_ratio = math.log(cfg.sigma_max / cfg.sigma_min)
    return T * (cfg.sigma_max ** 2 - cfg.sigma_min ** 2) / (2.0 * log_ratio)


# ──────────────────────────────────────────────────────────────────────────────
# VP / sub-VP helpers
# ──────────────────────────────────────────────────────────────────────────────

def _vp_cumulative_beta(T: float, cfg) -> float:
    """B(T) = ∫_0^T β(t) dt for linear β schedule."""
    # β(t) = β_min + (β_max − β_min) · t / T_total
    T_total = cfg.T
    return cfg.beta_min * T + 0.5 * (cfg.beta_max - cfg.beta_min) * T ** 2 / T_total


def vp_marginal_params(T: float, cfg: VPConfig) -> Tuple[float, float]:
    """
    Returns (alpha, var) where
      alpha = e^{−B(T)/2}  (mean coefficient)
      var   = 1 − alpha²   (marginal variance | X_0)
    """
    BT = _vp_cumulative_beta(T, cfg)
    alpha = math.exp(-0.5 * BT)
    var = max(1.0 - alpha ** 2, 1e-10)
    return alpha, var


def subvp_marginal_params(T: float, cfg: SubVPConfig) -> Tuple[float, float]:
    """
    Returns (alpha, var) for sub-VP.

    The sub-VP diffusion coefficient is g(t) = sqrt(β(t) · (1 − e^{−B(t)})).
    Variance | X_0:
      v(T) = e^{−B(T)} ∫_0^T β(s)(e^{B(s)} − 1) ds
           = (1 − e^{−B(T)}) − B(T)·e^{−B(T)}
           = (1 − alpha²) − B(T)·alpha²
    """
    _vp = VPConfig(beta_min=cfg.beta_min, beta_max=cfg.beta_max, T=cfg.T)
    BT = _vp_cumulative_beta(T, _vp)
    alpha = math.exp(-0.5 * BT)
    var = max((1.0 - alpha ** 2) - BT * alpha ** 2, 1e-10)
    return alpha, var


# ──────────────────────────────────────────────────────────────────────────────
# Forward sampling + Malliavin weights
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def simulate_ve(
    X0: torch.Tensor,
    T: float,
    cfg: VEConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Direct reparameterised simulation for the VE SDE.

    Parameters
    ----------
    X0  : (n, d)  initial samples from p_data
    T   : float   time horizon to simulate to

    Returns
    -------
    X_T : (n, d)  terminal samples
    H   : (n, d)  Malliavin score weight  H = (X_0 − X_T) / Σ²(T)
    """
    Sigma2 = ve_marginal_var(T, cfg)
    X_T = X0 + math.sqrt(Sigma2) * torch.randn_like(X0)
    H = (X0 - X_T) / Sigma2
    return X_T, H


@torch.no_grad()
def simulate_vp(
    X0: torch.Tensor,
    T: float,
    cfg: VPConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Direct reparameterised simulation for the VP SDE.

    X_T | X_0 ~ N(alpha·X_0, var·I)
    H = (alpha·X_0 − X_T) / var
    """
    alpha, var = vp_marginal_params(T, cfg)
    X_T = alpha * X0 + math.sqrt(var) * torch.randn_like(X0)
    H = (alpha * X0 - X_T) / var
    return X_T, H


@torch.no_grad()
def simulate_subvp(
    X0: torch.Tensor,
    T: float,
    cfg: SubVPConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Direct reparameterised simulation for the sub-VP SDE.

    X_T | X_0 ~ N(alpha·X_0, var·I)
    H = (alpha·X_0 − X_T) / var
    """
    alpha, var = subvp_marginal_params(T, cfg)
    X_T = alpha * X0 + math.sqrt(var) * torch.randn_like(X0)
    H = (alpha * X0 - X_T) / var
    return X_T, H


def simulate_linear(
    X0: torch.Tensor,
    T: float,
    sde_config,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dispatch to the appropriate linear SDE simulator."""
    if isinstance(sde_config, VEConfig):
        return simulate_ve(X0, T, sde_config)
    if isinstance(sde_config, VPConfig):
        return simulate_vp(X0, T, sde_config)
    if isinstance(sde_config, SubVPConfig):
        return simulate_subvp(X0, T, sde_config)
    raise ValueError(f"Unknown SDE config type: {type(sde_config)!r}")


# ──────────────────────────────────────────────────────────────────────────────
# Reverse SDE samplers (Euler–Maruyama)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def reverse_sample_ve(
    model,
    n_samples: int,
    cfg: VEConfig,
    n_steps: int = 250,
    clamp_score: float = 50.0,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Euler–Maruyama reverse SDE for VE:

        dX = σ²(t) · s(X, t) dt + σ(t) dW̄    (time running T → 0)

    Initialises from  X_T ~ N(0, Σ²(T)·I).

    Returns
    -------
    samples : (n_samples, 2) on CPU
    """
    T = cfg.T
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)

    Sigma2_T = ve_marginal_var(T, cfg)
    x = math.sqrt(Sigma2_T) * torch.randn(n_samples, 2, device=device)

    for k in reversed(range(n_steps)):
        t_curr = (k + 0.5) * dt
        t_tensor = torch.full((n_samples,), t_curr, device=device)

        sigma_t = ve_sigma_at(t_curr, cfg)

        score = model(t_tensor, x)
        score = torch.nan_to_num(score, nan=0.0).clamp(-clamp_score, clamp_score)

        x = x + sigma_t ** 2 * score * dt + sigma_t * sqrt_dt * torch.randn_like(x)
        x = torch.nan_to_num(x, nan=0.0)

    return x.cpu()


@torch.no_grad()
def reverse_sample_vp(
    model,
    n_samples: int,
    cfg: VPConfig,
    n_steps: int = 250,
    clamp_score: float = 50.0,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Euler–Maruyama reverse SDE for VP:

        dX = [½β(t)X + β(t)·s(X,t)] dt + √β(t) dW̄

    Initialises from  X_T ~ N(0, I).
    """
    T = cfg.T
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)

    x = torch.randn(n_samples, 2, device=device)

    for k in reversed(range(n_steps)):
        t_curr = (k + 0.5) * dt
        t_tensor = torch.full((n_samples,), t_curr, device=device)

        beta_t = cfg.beta_min + (cfg.beta_max - cfg.beta_min) * t_curr / T

        score = model(t_tensor, x)
        score = torch.nan_to_num(score, nan=0.0).clamp(-clamp_score, clamp_score)

        reverse_drift = 0.5 * beta_t * x + beta_t * score
        x = x + reverse_drift * dt + math.sqrt(beta_t) * sqrt_dt * torch.randn_like(x)
        x = torch.nan_to_num(x, nan=0.0)

    return x.cpu()


@torch.no_grad()
def reverse_sample_subvp(
    model,
    n_samples: int,
    cfg: SubVPConfig,
    n_steps: int = 250,
    clamp_score: float = 50.0,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Euler–Maruyama reverse SDE for sub-VP.

    Diffusion coefficient: g(t) = sqrt(β(t) · (1 − e^{−B(t)}))
    Reverse drift: ½β(t)X + g²(t)·s(X,t)
    """
    T = cfg.T
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)

    _vp = VPConfig(beta_min=cfg.beta_min, beta_max=cfg.beta_max, T=cfg.T)
    x = torch.randn(n_samples, 2, device=device)

    for k in reversed(range(n_steps)):
        t_curr = (k + 0.5) * dt
        t_tensor = torch.full((n_samples,), t_curr, device=device)

        beta_t = cfg.beta_min + (cfg.beta_max - cfg.beta_min) * t_curr / T
        Bt = _vp_cumulative_beta(t_curr, _vp)
        g2 = beta_t * max(1.0 - math.exp(-Bt), 1e-10)
        g = math.sqrt(g2)

        score = model(t_tensor, x)
        score = torch.nan_to_num(score, nan=0.0).clamp(-clamp_score, clamp_score)

        reverse_drift = 0.5 * beta_t * x + g2 * score
        x = x + reverse_drift * dt + g * sqrt_dt * torch.randn_like(x)
        x = torch.nan_to_num(x, nan=0.0)

    return x.cpu()


def reverse_sample_linear(
    model,
    n_samples: int,
    sde_config,
    n_steps: int = 250,
    clamp_score: float = 50.0,
    device: str = "cpu",
) -> torch.Tensor:
    """Dispatch to the appropriate reverse sampler."""
    if isinstance(sde_config, VEConfig):
        return reverse_sample_ve(model, n_samples, sde_config,
                                 n_steps=n_steps, clamp_score=clamp_score, device=device)
    if isinstance(sde_config, VPConfig):
        return reverse_sample_vp(model, n_samples, sde_config,
                                 n_steps=n_steps, clamp_score=clamp_score, device=device)
    if isinstance(sde_config, SubVPConfig):
        return reverse_sample_subvp(model, n_samples, sde_config,
                                    n_steps=n_steps, clamp_score=clamp_score, device=device)
    raise ValueError(f"Unknown SDE config type: {type(sde_config)!r}")
