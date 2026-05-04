"""
2D dataset generators for the Mirafzali reproduction experiments.

Datasets
--------
- 8-GMM        (re-exported from sde_2d for convenience)
- Checkerboard
- Swiss Roll (2D spiral)
- Single Swiss Roll (legacy single-spiral; kept for backward compatibility)
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

def sample_single_swissroll(n: int, noise: float = 0.05, device: str = "cpu"):
    """
    Single 2D Swiss Roll — a planar spiral (legacy implementation).

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


# 5-component Swiss Roll mixture (Mirafzali et al. Figure 1/2)
# Centers arranged on a regular pentagon of radius _SR_RADIUS.
# Each component is a small 2-D spiral scaled to _SR_ARM_SCALE,
# with additive noise _SR_NOISE.

_SR_N_ROLLS    = 5
_SR_RADIUS     = 2.5          # distance of each spiral centre from origin
_SR_ARM_SCALE  = 0.45         # half-width of each individual spiral arm
_SR_T_MIN      = 1.5 * math.pi
_SR_T_MAX      = 4.5 * math.pi  # shorter arms than the single roll
_SR_NOISE      = 0.04


def sample_swissroll(n: int, noise: float = _SR_NOISE, device: str = "cpu"):
    """
    5-component Swiss Roll mixture, following Mirafzali et al. Figure 1/2.

    Five small 2-D spirals are placed at the vertices of a regular pentagon
    of radius ``_SR_RADIUS``.  Each spiral is uniformly sampled along its
    arc and independently rotated to reduce visual overlap.

    Parameters
    ----------
    n     : total number of samples (divided equally across rolls)
    noise : additive Gaussian std (in normalised spiral coordinates)
    device : torch device string

    Returns
    -------
    x : (n, 2) samples on the 5-roll distribution
    """
    angles_center = [2 * math.pi * k / _SR_N_ROLLS for k in range(_SR_N_ROLLS)]
    # Independent rotation of each spiral arm so they don't align
    rot_offsets   = [2 * math.pi * k / _SR_N_ROLLS for k in range(_SR_N_ROLLS)]

    # Distribute n as evenly as possible across rolls
    base, rem = divmod(n, _SR_N_ROLLS)
    counts = [base + (1 if k < rem else 0) for k in range(_SR_N_ROLLS)]

    parts = []
    for k in range(_SR_N_ROLLS):
        nk = counts[k]
        if nk == 0:
            continue

        # Centre of this spiral
        cx = _SR_RADIUS * math.cos(angles_center[k])
        cy = _SR_RADIUS * math.sin(angles_center[k])

        # Sample arc parameter
        t = _SR_T_MIN + (_SR_T_MAX - _SR_T_MIN) * torch.rand(nk, device=device)

        # Local spiral (normalised to ≈ [−1, 1])
        scale = _SR_T_MAX * 0.5
        lx = t * torch.cos(t + rot_offsets[k]) / scale
        ly = t * torch.sin(t + rot_offsets[k]) / scale

        # Scale to arm width and translate to centre
        px = cx + _SR_ARM_SCALE * lx
        py = cy + _SR_ARM_SCALE * ly

        if noise > 0:
            px = px + noise * torch.randn(nk, device=device)
            py = py + noise * torch.randn(nk, device=device)

        parts.append(torch.stack([px, py], dim=1))

    return torch.cat(parts, dim=0)


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

_SAMPLERS = {
    "8gmm":           sample_8gmm,
    "checkerboard":   sample_checkerboard,
    "swissroll":      sample_swissroll,
    "single_swissroll": sample_single_swissroll,
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
