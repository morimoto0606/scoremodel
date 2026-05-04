"""
2D dataset generators for the Mirafzali reproduction experiments.

Datasets
--------
- 8-GMM        (re-exported from sde_2d for convenience)
- Checkerboard
- Swiss Roll (2D spiral)
"""

import math

import torch


# ──────────────────────────────────────────────────────────────────────────────
# 8-GMM
# ──────────────────────────────────────────────────────────────────────────────

def sample_8gmm(n: int, radius: float = 2.0, std: float = 0.08,
                device: str = "cpu"):
    """
    Sample from an 8-component Gaussian mixture arranged on a circle.

    Returns
    -------
    x       : (n, 2) samples
    centers : (8, 2) mode centres
    """
    angles = torch.arange(8, device=device) * (2 * math.pi / 8)
    centers = radius * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    idx = torch.randint(0, 8, (n,), device=device)
    x = centers[idx] + std * torch.randn(n, 2, device=device)
    return x, centers


# ──────────────────────────────────────────────────────────────────────────────
# Checkerboard
# ──────────────────────────────────────────────────────────────────────────────

def sample_checkerboard(n: int, half_width: float = 2.0, n_cells: int = 4,
                        device: str = "cpu"):
    """
    Uniform samples from the active (even-parity) cells of an
    n_cells × n_cells checkerboard in [−half_width, half_width]².

    Cell (i, j) is active iff (i + j) % 2 == 0.

    Returns
    -------
    x : (n, 2)
    """
    active = [(i, j) for i in range(n_cells) for j in range(n_cells)
              if (i + j) % 2 == 0]
    n_active = len(active)
    cell_size = 2.0 * half_width / n_cells

    choice = torch.randint(0, n_active, (n,), device=device)
    rows = torch.tensor([c[0] for c in active], dtype=torch.float32, device=device)[choice]
    cols = torch.tensor([c[1] for c in active], dtype=torch.float32, device=device)[choice]

    u = torch.rand(n, 2, device=device)
    x = -half_width + (rows + u[:, 0]) * cell_size
    y = -half_width + (cols + u[:, 1]) * cell_size
    return torch.stack([x, y], dim=1)


# ──────────────────────────────────────────────────────────────────────────────
# Swiss Roll (2D spiral)
# ──────────────────────────────────────────────────────────────────────────────

def sample_swissroll(n: int, noise: float = 0.05, device: str = "cpu"):
    """
    2D Swiss Roll — a planar spiral.

    t ~ Uniform[3π/2, 15π/2]
    (x, y) = t * (cos t, sin t)  then normalised so the bounding box
    is roughly [−2, 2]².

    Parameters
    ----------
    noise : additive Gaussian std after normalisation
    """
    t_min = 1.5 * math.pi
    t_max = 7.5 * math.pi
    t = t_min + (t_max - t_min) * torch.rand(n, device=device)

    raw_x = t * torch.cos(t)
    raw_y = t * torch.sin(t)

    # Normalise to ≈ [−2, 2]²
    scale = t_max * 0.5          # rough max radius
    x = raw_x / scale * 2.0
    y = raw_y / scale * 2.0

    if noise > 0:
        x = x + noise * torch.randn(n, device=device)
        y = y + noise * torch.randn(n, device=device)

    return torch.stack([x, y], dim=1)


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

_SAMPLERS = {
    "8gmm":        sample_8gmm,
    "checkerboard": sample_checkerboard,
    "swissroll":   sample_swissroll,
}


def get_sampler(name: str):
    """
    Return the sampler callable for *name*.

    The callable has signature  f(n, device="cpu")  and returns a tensor
    of shape (n, 2).  For 8gmm it also returns centers; callers that only
    need samples should index position [0].
    """
    if name not in _SAMPLERS:
        raise ValueError(f"Unknown dataset {name!r}. Choose from {list(_SAMPLERS)}")
    return _SAMPLERS[name]
