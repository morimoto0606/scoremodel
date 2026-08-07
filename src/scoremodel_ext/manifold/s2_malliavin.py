"""De Bortoli-style S2 forward diffusion with a Malliavin teacher.

The forward integrator is a geodesic random walk on the base manifold.  A
path is a differentiable function of standard ambient Gaussian increments,
which lets :mod:`malliavin_teacher` compute the full discrete Skorokhod
correction without using the additive-noise formula of Mirafzali et al.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .malliavin_teacher import (
    DiscreteMalliavinTeacher,
    batched_discrete_malliavin_skorokhod_teacher,
    discrete_malliavin_skorokhod_teacher,
    tangent_malliavin_skorokhod_teacher,
)
from .beta_schedule import LegacyUnitBetaSchedule, LinearBetaSchedule


Tensor = torch.Tensor


def _batched_s2_projector(x: Tensor) -> Tensor:
    """Batched ``I - x x^T`` for tensors with final dimension three."""

    x = x / torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    eye = torch.eye(3, dtype=x.dtype, device=x.device)
    return eye.expand(*x.shape[:-1], 3, 3) - x[..., :, None] * x[..., None, :]


class S2SkorokhodScoreModel(nn.Module):
    r"""Turn a network predicting ``E[delta | X_t]`` into a tangent score.

    For the projected ambient fields on ``S2``, reconstructing the vector
    score cancels the individual field-divergence terms:

    .. math::

        s(t,x)=-P_x\,\mathbb E[\delta\mid X_t=x].

    The wrapped network must follow the existing Mirafzali interface
    ``network(t, x) -> R3``.
    """

    def __init__(self, skorokhod_network: nn.Module):
        super().__init__()
        self.skorokhod_network = skorokhod_network

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        delta = self.skorokhod_network(t, x)
        projector = _batched_s2_projector(x)
        return -torch.einsum("...ij,...j->...i", projector, delta)


def s2_projector(x: Tensor) -> Tensor:
    """Orthogonal projector from R3 onto ``T_x S2``."""

    x = x.reshape(3)
    eye = torch.eye(3, dtype=x.dtype, device=x.device)
    return eye - torch.outer(x, x)


def s2_to_tangent(vector: Tensor, base_point: Tensor) -> Tensor:
    """Project an ambient vector onto the sphere tangent space."""

    return s2_projector(base_point) @ vector.reshape(3)


def s2_exp(base_point: Tensor, tangent_vector: Tensor) -> Tensor:
    """Stable exponential map on the unit two-sphere."""

    base_point = base_point.reshape(3)
    tangent_vector = s2_to_tangent(tangent_vector, base_point)
    norm = torch.linalg.vector_norm(tangent_vector)
    # torch.sinc(q) = sin(pi q)/(pi q), including its stable value at zero.
    sinc = torch.sinc(norm / math.pi)
    endpoint = torch.cos(norm) * base_point + sinc * tangent_vector
    # The formula is norm preserving analytically.  Normalisation controls
    # accumulated floating-point drift while remaining differentiable.
    return endpoint / torch.linalg.vector_norm(endpoint)


def s2_grw_endpoint(
    initial_point: Tensor,
    standard_noise: Tensor,
    terminal_time: float,
) -> Tensor:
    r"""Simulate the endpoint of a base-manifold geodesic random walk.

    Each ``standard_noise[k]`` is an ambient ``N(0,I_3)`` vector.  Projection
    produces the correct isotropic tangent covariance and

    .. math::

        X_{k+1}=\operatorname{Exp}_{X_k}
        \left(\sqrt{\Delta t}\,P_{X_k}Z_k\right)

    converges to Brownian motion with generator ``(1/2) Delta_S2``.
    """

    if standard_noise.ndim != 2 or standard_noise.shape[-1] != 3:
        raise ValueError("standard_noise must have shape [n_steps, 3]")
    if standard_noise.shape[0] < 1:
        raise ValueError("at least one GRW step is required")
    if terminal_time <= 0:
        raise ValueError("terminal_time must be positive")

    x = initial_point.reshape(3)
    x = x / torch.linalg.vector_norm(x)
    sqrt_dt = math.sqrt(terminal_time / standard_noise.shape[0])
    for increment in standard_noise:
        tangent_increment = sqrt_dt * s2_to_tangent(increment, x)
        x = s2_exp(x, tangent_increment)
    return x


def s2_grw_endpoint_tensor_time(
    initial_point: Tensor,
    standard_noise: Tensor,
    terminal_time: Tensor,
) -> Tensor:
    """Pure tensor-time GRW used inside ``torch.func`` transforms.

    Shape/value validation belongs to the outer batched adapter.  Keeping this
    single-sample function free of tensor-dependent Python branches and scalar
    extraction makes it composable with nested ``jacrev`` and ``vmap``.
    """

    x = initial_point.reshape(3)
    x = x / torch.linalg.vector_norm(x)
    sqrt_dt = torch.sqrt(terminal_time / standard_noise.shape[0])
    for step in range(standard_noise.shape[0]):
        increment = standard_noise[step]
        tangent_increment = sqrt_dt * s2_to_tangent(increment, x)
        x = s2_exp(x, tangent_increment)
    return x


def s2_tangent_basis(endpoint: Tensor) -> Tensor:
    """Orthonormal tangent basis for the sphere at ``endpoint``."""

    endpoint = endpoint.reshape(3)
    if torch.linalg.vector_norm(endpoint).item() < 1e-12:
        raise ValueError("endpoint must be non-zero")
    endpoint = endpoint / torch.linalg.vector_norm(endpoint)
    tangent = torch.tensor([1.0, 0.0, 0.0], dtype=endpoint.dtype, device=endpoint.device)
    if abs(float(torch.dot(endpoint, tangent))) > 0.9:
        tangent = torch.tensor([0.0, 1.0, 0.0], dtype=endpoint.dtype, device=endpoint.device)
    e1 = tangent - torch.dot(tangent, endpoint) * endpoint
    e1 = e1 / torch.linalg.vector_norm(e1)
    e2 = torch.linalg.cross(endpoint, e1)
    return torch.stack((e1, e2), dim=1)


def s2_reconstruct_score_vector(directional_scores: Tensor, endpoint: Tensor) -> Tensor:
    """Reconstruct a tangent score from tangent-basis or projected-field weights."""

    endpoint = endpoint.reshape(3)
    directional_scores = directional_scores.reshape(-1)
    basis = s2_tangent_basis(endpoint)
    if directional_scores.numel() == basis.shape[1]:
        return basis @ directional_scores
    fields = s2_projected_ambient_fields(endpoint)
    return torch.linalg.pinv(fields.transpose(0, 1), rtol=1e-7) @ directional_scores


def s2_projected_ambient_fields(endpoint: Tensor) -> Tensor:
    """Three redundant fields ``V_j(x)=P_x e_j`` spanning ``T_x S2``."""

    return s2_projector(endpoint)


def s2_projected_ambient_field_divergence(endpoint: Tensor) -> Tensor:
    r"""Divergences of ``V_j(x)=P_x e_j`` with respect to sphere volume.

    Since ``V_j = grad_S2 x_j`` and ``Delta_S2 x_j = -2 x_j``, one has
    ``div_S2 V_j = -2 x_j``.
    """

    return -2.0 * endpoint.reshape(3)


def s2_discrete_malliavin_teacher(
    initial_point: Tensor,
    standard_noise: Tensor,
    terminal_time: float,
    *,
    covariance_regularization: float = 1e-6,
    vectorize_jacobian: bool = True,
) -> DiscreteMalliavinTeacher:
    """Compute one full Skorokhod-corrected S2 path teacher."""

    endpoint_fn = lambda z: s2_grw_endpoint(initial_point, z, terminal_time)
    return discrete_malliavin_skorokhod_teacher(
        endpoint_fn,
        standard_noise,
        s2_projected_ambient_fields,
        field_divergence_fn=s2_projected_ambient_field_divergence,
        covariance_regularization=covariance_regularization,
        vectorize_jacobian=vectorize_jacobian,
    )


def s2_batched_discrete_malliavin_teacher(
    initial_points: Tensor,
    standard_noises: Tensor,
    terminal_times: Tensor,
    *,
    covariance_regularization: float = 1e-6,
) -> DiscreteMalliavinTeacher:
    """Compute ordered S2 teachers with one sample-direction ``vmap``."""

    if initial_points.ndim != 2 or initial_points.shape[1] != 3:
        raise ValueError("initial_points must have shape [batch, 3]")
    if standard_noises.ndim != 3 or standard_noises.shape[2] != 3:
        raise ValueError("standard_noises must have shape [batch, n_steps, 3]")
    if standard_noises.shape[1] < 1:
        raise ValueError("at least one GRW step is required")
    if terminal_times.shape != (initial_points.shape[0],):
        raise ValueError("terminal_times must have shape [batch]")
    if standard_noises.shape[0] != initial_points.shape[0]:
        raise ValueError("standard_noises batch dimension does not match")
    if bool((terminal_times <= 0).any().detach().cpu()):
        raise ValueError("all terminal_times must be positive")
    return batched_discrete_malliavin_skorokhod_teacher(
        s2_grw_endpoint_tensor_time,
        initial_points,
        standard_noises,
        terminal_times,
        s2_projected_ambient_fields,
        field_divergence_fn=s2_projected_ambient_field_divergence,
        covariance_regularization=covariance_regularization,
    )


def s2_tangent_malliavin_teacher(
    initial_point: Tensor,
    standard_noise: Tensor,
    terminal_time: float,
    *,
    field_index: Optional[int] = None,
    covariance_regularization: float = 1e-6,
    vectorize_jacobian: bool = True,
) -> DiscreteMalliavinTeacher:
    """Compute a tangent-space Malliavin teacher on S2.

    When ``field_index`` is omitted, the teacher uses the three projected
    ambient coordinate fields ``P_x e_i`` simultaneously.
    """

    if field_index is not None and not 0 <= field_index < 3:
        raise ValueError("field_index must be in {0,1,2}")

    endpoint_fn = lambda z: s2_grw_endpoint(initial_point, z, terminal_time)
    tangent_basis_fn = lambda endpoint: s2_tangent_basis(endpoint)

    def target_fields_fn(endpoint: Tensor) -> Tensor:
        endpoint = endpoint.reshape(3)
        projected_fields = s2_projector(endpoint)
        if field_index is None:
            return projected_fields
        return projected_fields[:, field_index]

    def field_divergence_fn(endpoint: Tensor) -> Tensor:
        endpoint = endpoint.reshape(3)
        divergences = -2.0 * endpoint
        if field_index is None:
            return divergences
        return divergences[field_index]

    return tangent_malliavin_skorokhod_teacher(
        endpoint_fn,
        standard_noise,
        tangent_basis_fn,
        target_fields_fn,
        field_divergence_fn=field_divergence_fn,
        covariance_regularization=covariance_regularization,
        vectorize_jacobian=vectorize_jacobian,
    )


def s2_heat_kernel_score(
    initial_point: Tensor,
    endpoint: Tensor,
    terminal_time: float,
    *,
    n_terms: int = 80,
) -> Tensor:
    r"""Spectral heat-kernel score on S2 for generator ``(1/2) Delta``.

    The transition density is

    .. math::

        p_t(x_0,x)=\frac{1}{4\pi}\sum_{l=0}^{\infty}
        (2l+1)e^{-l(l+1)t/2}P_l(x_0^\top x).

    Both the Legendre polynomials and their derivatives are evaluated by a
    recurrence, avoiding differentiation through ``acos`` near the diagonal.
    For very small times this alternating spectral sum needs more terms and
    higher precision; use the Varadhan score below as a separate small-time
    diagnostic rather than silently switching the reference target.
    """

    if terminal_time <= 0:
        raise ValueError("terminal_time must be positive")
    if n_terms < 2:
        raise ValueError("n_terms must be at least two")

    x0 = initial_point.reshape(3)
    x0 = x0 / torch.linalg.vector_norm(x0)
    x = endpoint.reshape(3)
    x = x / torch.linalg.vector_norm(x)
    cosine = torch.clamp(torch.dot(x0, x), -1.0, 1.0)

    p_prev = torch.ones((), dtype=x.dtype, device=x.device)
    p_curr = cosine
    dp_prev = torch.zeros_like(cosine)
    dp_curr = torch.ones_like(cosine)

    density = torch.ones_like(cosine) / (4.0 * math.pi)
    density_derivative = torch.zeros_like(cosine)

    weight_1 = 3.0 * math.exp(-terminal_time) / (4.0 * math.pi)
    density = density + weight_1 * p_curr
    density_derivative = density_derivative + weight_1 * dp_curr

    for degree in range(2, n_terms):
        degree_float = float(degree)
        p_next = (
            (2.0 * degree_float - 1.0) * cosine * p_curr
            - (degree_float - 1.0) * p_prev
        ) / degree_float
        dp_next = (
            (2.0 * degree_float - 1.0) * (p_curr + cosine * dp_curr)
            - (degree_float - 1.0) * dp_prev
        ) / degree_float
        weight = (
            (2.0 * degree_float + 1.0)
            * math.exp(-0.5 * degree_float * (degree_float + 1.0) * terminal_time)
            / (4.0 * math.pi)
        )
        density = density + weight * p_next
        density_derivative = density_derivative + weight * dp_next
        p_prev, p_curr = p_curr, p_next
        dp_prev, dp_curr = dp_curr, dp_next

    if bool((density <= 0).detach().cpu()):
        raise FloatingPointError(
            "truncated heat-kernel series is non-positive; increase n_terms "
            "or use a later terminal_time"
        )
    grad_cosine = x0 - cosine * x
    return (density_derivative / density) * grad_cosine


def _s2_heat_kernel_score_tensor_time(
    initial_point: Tensor,
    endpoint: Tensor,
    terminal_time: Tensor,
    *,
    n_terms: int,
) -> Tuple[Tensor, Tensor]:
    """Pure tensor-time form of the existing truncated Heat score.

    Validation is deliberately performed by the outer batched adapter.  This
    function contains no scalar extraction or tensor-dependent Python branch,
    so it can be transformed with ``torch.func.vmap``.
    """

    x0 = initial_point.reshape(3)
    x0 = x0 / torch.linalg.vector_norm(x0)
    x = endpoint.reshape(3)
    x = x / torch.linalg.vector_norm(x)
    cosine = torch.clamp(torch.dot(x0, x), -1.0, 1.0)

    p_prev = torch.ones((), dtype=x.dtype, device=x.device)
    p_curr = cosine
    dp_prev = torch.zeros_like(cosine)
    dp_curr = torch.ones_like(cosine)

    density = torch.ones_like(cosine) / (4.0 * math.pi)
    density_derivative = torch.zeros_like(cosine)
    weight_1 = 3.0 * torch.exp(-terminal_time) / (4.0 * math.pi)
    density = density + weight_1 * p_curr
    density_derivative = density_derivative + weight_1 * dp_curr

    for degree in range(2, n_terms):
        degree_float = float(degree)
        p_next = (
            (2.0 * degree_float - 1.0) * cosine * p_curr
            - (degree_float - 1.0) * p_prev
        ) / degree_float
        dp_next = (
            (2.0 * degree_float - 1.0) * (p_curr + cosine * dp_curr)
            - (degree_float - 1.0) * dp_prev
        ) / degree_float
        weight = (
            (2.0 * degree_float + 1.0)
            * torch.exp(
                -0.5
                * degree_float
                * (degree_float + 1.0)
                * terminal_time
            )
            / (4.0 * math.pi)
        )
        density = density + weight * p_next
        density_derivative = density_derivative + weight * dp_next
        p_prev, p_curr = p_curr, p_next
        dp_prev, dp_curr = dp_curr, dp_next

    grad_cosine = x0 - cosine * x
    return (density_derivative / density) * grad_cosine, density


def s2_batched_heat_kernel_score(
    initial_points: Tensor,
    endpoints: Tensor,
    terminal_times: Tensor,
    *,
    n_terms: int = 80,
) -> Tensor:
    """Evaluate the unchanged Heat teacher for a sample batch."""

    if initial_points.ndim != 2 or initial_points.shape[1] != 3:
        raise ValueError("initial_points must have shape [batch, 3]")
    if endpoints.shape != initial_points.shape:
        raise ValueError("endpoints must have the same shape as initial_points")
    if terminal_times.shape != (initial_points.shape[0],):
        raise ValueError("terminal_times must have shape [batch]")
    if n_terms < 2:
        raise ValueError("n_terms must be at least two")
    if bool((terminal_times <= 0).any().detach().cpu()):
        raise ValueError("all terminal_times must be positive")

    score, density = torch.func.vmap(
        lambda x0, x, t: _s2_heat_kernel_score_tensor_time(
            x0,
            x,
            t,
            n_terms=n_terms,
        )
    )(initial_points, endpoints, terminal_times)
    if bool((density <= 0).any().detach().cpu()):
        raise FloatingPointError(
            "truncated heat-kernel series is non-positive; increase n_terms "
            "or use a later terminal_time"
        )
    return score


def s2_batched_grw_endpoint(
    initial_points: Tensor,
    standard_noises: Tensor,
    terminal_times: Tensor,
) -> Tensor:
    """Batch the existing tensor-safe single-sample GRW endpoint map."""

    if initial_points.ndim != 2 or initial_points.shape[1] != 3:
        raise ValueError("initial_points must have shape [batch, 3]")
    if standard_noises.ndim != 3 or standard_noises.shape[2] != 3:
        raise ValueError("standard_noises must have shape [batch, n_steps, 3]")
    if standard_noises.shape[0] != initial_points.shape[0]:
        raise ValueError("standard_noises batch dimension does not match")
    if standard_noises.shape[1] < 1:
        raise ValueError("at least one GRW step is required")
    if terminal_times.shape != (initial_points.shape[0],):
        raise ValueError("terminal_times must have shape [batch]")
    if bool((terminal_times <= 0).any().detach().cpu()):
        raise ValueError("all terminal_times must be positive")
    return torch.func.vmap(s2_grw_endpoint_tensor_time)(
        initial_points,
        standard_noises,
        terminal_times,
    )


def s2_varadhan_score(
    initial_point: Tensor,
    endpoint: Tensor,
    terminal_time: float,
) -> Tensor:
    """Small-time score ``Log_endpoint(initial_point) / terminal_time``."""

    if terminal_time <= 0:
        raise ValueError("terminal_time must be positive")
    x0 = initial_point.reshape(3)
    x0 = x0 / torch.linalg.vector_norm(x0)
    x = endpoint.reshape(3)
    x = x / torch.linalg.vector_norm(x)
    cosine = torch.clamp(torch.dot(x, x0), -1.0, 1.0)
    angle = torch.acos(cosine)
    tangent = x0 - cosine * x
    tangent_norm = torch.linalg.vector_norm(tangent)
    scale = torch.where(
        tangent_norm > 1e-10,
        angle / tangent_norm,
        torch.ones_like(tangent_norm),
    )
    return scale * tangent / terminal_time


def sample_s2_teacher_path(
    initial_point: Tensor,
    *,
    terminal_time: float,
    n_steps: int,
    covariance_regularization: float = 1e-6,
    generator: torch.Generator | None = None,
    vectorize_jacobian: bool = True,
) -> Tuple[Tensor, DiscreteMalliavinTeacher]:
    """Draw standard noise and return it together with its path teacher."""

    noise = torch.randn(
        n_steps,
        3,
        dtype=initial_point.dtype,
        device=initial_point.device,
        generator=generator,
    )
    teacher = s2_discrete_malliavin_teacher(
        initial_point,
        noise,
        terminal_time,
        covariance_regularization=covariance_regularization,
        vectorize_jacobian=vectorize_jacobian,
    )
    return noise, teacher


def s2_reverse_grw(
    terminal_points: Tensor,
    score_fn,
    *,
    terminal_time: float,
    n_steps: int,
    standard_noise: Tensor | None = None,
    minimum_forward_time: float = 1e-3,
    return_first_step: bool = False,
    debug_output_dir: str | Path | None = None,
    beta_schedule: LegacyUnitBetaSchedule | LinearBetaSchedule | None = None,
) -> Tensor | Tuple[Tensor, Tensor]:
    r"""De Bortoli reverse GRW for a Brownian forward process on ``S2``.

    For forward generator ``(1/2) Delta`` the reverse drift is the manifold
    score.  The update is

    .. math::

        Y_{k+1}=\operatorname{Exp}_{Y_k}\left(
        \Delta t\,s(T-\tau_k,Y_k)+\sqrt{\Delta t}\,P_{Y_k}Z_k
        \right).

    Parameters
    ----------
    score_fn:
        Callable ``score_fn(t, x_batch)`` returning ambient tangent vectors.
    standard_noise:
        Optional tensor ``[n_steps, batch, 3]``.  Supplying it makes server
        comparisons reproducible.
    return_first_step:
        If true, also return the samples immediately after the first update.
        The default return value and numerical path are unchanged.
    debug_output_dir:
        Optional directory for step-0/step-1 intermediate tensors. Debugging is
        opt-in; leaving this unset preserves the ordinary numerical path.
    beta_schedule:
        Optional physical-time Brownian beta schedule. ``None`` and
        :class:`LegacyUnitBetaSchedule` use the unchanged legacy numerical
        path. A linear schedule uses exact interval Brownian times while the
        score network continues to receive physical time.
    """

    if beta_schedule is None or isinstance(beta_schedule, LegacyUnitBetaSchedule):
        return _s2_reverse_grw_legacy(
            terminal_points,
            score_fn,
            terminal_time=terminal_time,
            n_steps=n_steps,
            standard_noise=standard_noise,
            minimum_forward_time=minimum_forward_time,
            return_first_step=return_first_step,
            debug_output_dir=debug_output_dir,
        )
    return _s2_reverse_grw_linear_schedule(
        terminal_points,
        score_fn,
        terminal_time=terminal_time,
        n_steps=n_steps,
        standard_noise=standard_noise,
        minimum_forward_time=minimum_forward_time,
        return_first_step=return_first_step,
        debug_output_dir=debug_output_dir,
        beta_schedule=beta_schedule,
    )


def _s2_reverse_grw_legacy(
    terminal_points: Tensor,
    score_fn,
    *,
    terminal_time: float,
    n_steps: int,
    standard_noise: Tensor | None,
    minimum_forward_time: float,
    return_first_step: bool,
    debug_output_dir: str | Path | None,
) -> Tensor | Tuple[Tensor, Tensor]:
    """Original beta=1 reverse implementation, kept numerically unchanged."""

    if terminal_points.ndim != 2 or terminal_points.shape[1] != 3:
        raise ValueError("terminal_points must have shape [batch, 3]")
    if terminal_time <= 0 or n_steps < 1:
        raise ValueError("terminal_time and n_steps must be positive")

    points = terminal_points / torch.linalg.vector_norm(
        terminal_points, dim=1, keepdim=True
    )
    if standard_noise is None:
        standard_noise = torch.randn(
            n_steps,
            points.shape[0],
            3,
            dtype=points.dtype,
            device=points.device,
        )
    expected_shape = (n_steps, points.shape[0], 3)
    if standard_noise.shape != expected_shape:
        raise ValueError(f"standard_noise must have shape {expected_shape}")

    dt = terminal_time / n_steps
    sqrt_dt = math.sqrt(dt)
    debug_dir = Path(debug_output_dir) if debug_output_dir is not None else None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
    first_step_points = None
    for step in range(n_steps):
        input_points = points
        forward_time = max(terminal_time - step * dt, minimum_forward_time)
        time_batch = torch.full(
            (points.shape[0],),
            forward_time,
            dtype=points.dtype,
            device=points.device,
        )
        raw_score = score_fn(time_batch, points)
        projector = _batched_s2_projector(points)
        projected_score = torch.einsum("bij,bj->bi", projector, raw_score)
        raw_noise = standard_noise[step]
        projected_noise = torch.einsum(
            "bij,bj->bi", projector, standard_noise[step]
        )
        tangent_increment = dt * projected_score + sqrt_dt * projected_noise
        points = torch.stack(
            [s2_exp(point, increment) for point, increment in zip(points, tangent_increment)]
        )
        if debug_dir is not None and step < 2:
            torch.save(
                {
                    "input_points": input_points.detach().cpu(),
                    "forward_time": torch.tensor(forward_time, dtype=points.dtype),
                    "time_batch": time_batch.detach().cpu(),
                    "raw_score": raw_score.detach().cpu(),
                    "projector": projector.detach().cpu(),
                    "projected_score": projected_score.detach().cpu(),
                    "raw_noise": raw_noise.detach().cpu(),
                    "projected_noise": projected_noise.detach().cpu(),
                    "dt": torch.tensor(dt, dtype=points.dtype),
                    "sqrt_dt": torch.tensor(sqrt_dt, dtype=points.dtype),
                    "tangent_increment": tangent_increment.detach().cpu(),
                    "output_points": points.detach().cpu(),
                },
                debug_dir / f"reverse_debug_step_{step:03d}.pt",
            )
        if return_first_step and step == 0:
            first_step_points = points.clone()
    if return_first_step:
        assert first_step_points is not None
        return points, first_step_points
    return points


def _s2_reverse_grw_linear_schedule(
    terminal_points: Tensor,
    score_fn,
    *,
    terminal_time: float,
    n_steps: int,
    standard_noise: Tensor | None,
    minimum_forward_time: float,
    return_first_step: bool,
    debug_output_dir: str | Path | None,
    beta_schedule: LinearBetaSchedule,
) -> Tensor | Tuple[Tensor, Tensor]:
    """Reverse GRW using exact Brownian time over each physical interval."""

    if terminal_points.ndim != 2 or terminal_points.shape[1] != 3:
        raise ValueError("terminal_points must have shape [batch, 3]")
    if terminal_time <= beta_schedule.t0 or n_steps < 1:
        raise ValueError(
            "terminal_time must exceed beta schedule t0 and n_steps must be positive"
        )
    if terminal_time > beta_schedule.tf:
        raise ValueError("terminal_time must not exceed beta schedule tf")

    points = terminal_points / torch.linalg.vector_norm(
        terminal_points, dim=1, keepdim=True
    )
    if standard_noise is None:
        standard_noise = torch.randn(
            n_steps,
            points.shape[0],
            3,
            dtype=points.dtype,
            device=points.device,
        )
    expected_shape = (n_steps, points.shape[0], 3)
    if standard_noise.shape != expected_shape:
        raise ValueError(f"standard_noise must have shape {expected_shape}")

    dt = (terminal_time - beta_schedule.t0) / n_steps
    debug_dir = Path(debug_output_dir) if debug_output_dir is not None else None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
    first_step_points = None
    for step in range(n_steps):
        input_points = points
        physical_time = terminal_time - step * dt
        next_physical_time = terminal_time - (step + 1) * dt
        forward_time = max(physical_time, minimum_forward_time)
        time_batch = torch.full(
            (points.shape[0],),
            forward_time,
            dtype=points.dtype,
            device=points.device,
        )
        raw_score = score_fn(time_batch, points)
        projector = _batched_s2_projector(points)
        projected_score = torch.einsum("bij,bj->bi", projector, raw_score)
        raw_noise = standard_noise[step]
        projected_noise = torch.einsum(
            "bij,bj->bi", projector, standard_noise[step]
        )
        delta_tau = beta_schedule.interval_brownian_time(
            next_physical_time,
            physical_time,
        )
        if delta_tau <= 0.0:
            raise ValueError("beta schedule produced non-positive Brownian time")
        sqrt_delta_tau = math.sqrt(delta_tau)
        tangent_increment = (
            delta_tau * projected_score + sqrt_delta_tau * projected_noise
        )
        points = torch.stack(
            [s2_exp(point, increment) for point, increment in zip(points, tangent_increment)]
        )
        if debug_dir is not None and step < 2:
            torch.save(
                {
                    "input_points": input_points.detach().cpu(),
                    "forward_time": torch.tensor(forward_time, dtype=points.dtype),
                    "time_batch": time_batch.detach().cpu(),
                    "raw_score": raw_score.detach().cpu(),
                    "projector": projector.detach().cpu(),
                    "projected_score": projected_score.detach().cpu(),
                    "raw_noise": raw_noise.detach().cpu(),
                    "projected_noise": projected_noise.detach().cpu(),
                    "dt": torch.tensor(dt, dtype=points.dtype),
                    "sqrt_dt": torch.tensor(sqrt_delta_tau, dtype=points.dtype),
                    "tangent_increment": tangent_increment.detach().cpu(),
                    "output_points": points.detach().cpu(),
                },
                debug_dir / f"reverse_debug_step_{step:03d}.pt",
            )
        if return_first_step and step == 0:
            first_step_points = points.clone()
    if return_first_step:
        assert first_step_points is not None
        return points, first_step_points
    return points
