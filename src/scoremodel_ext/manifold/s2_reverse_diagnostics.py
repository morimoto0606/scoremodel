"""Opt-in reverse-sampler diagnostics for the Earthquake S2 experiment.

This module intentionally lives outside the production reverse sampler.  It
provides a traced copy of the current update and an upstream-compatible update
so saved models, terminal points, and noises can be compared without changing
training or the ordinary :func:`s2_reverse_grw` numerical path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch

from .beta_schedule import LegacyUnitBetaSchedule, LinearBetaSchedule
from .s2_malliavin import _batched_s2_projector, s2_exp


Tensor = torch.Tensor
BetaSchedule = LegacyUnitBetaSchedule | LinearBetaSchedule | None
UPSTREAM_REVERSE_STEPS = 100
UPSTREAM_REVERSE_EPSILON = 1e-3


@dataclass(frozen=True)
class S2ReverseSamplerDiagnostics:
    """Tensor-valued trace returned by an opt-in diagnostic sampler."""

    final_samples: Tensor
    trajectory: Tensor
    time_grid: Tensor
    network_output_norm: Tensor
    score_norm: Tensor
    projected_score_norm: Tensor
    beta_score_norm: Tensor
    beta_projected_score_norm: Tensor
    drift_increment_norm: Tensor
    noise_increment_norm: Tensor
    score_std: Tensor

    def as_artifact(self) -> dict[str, Tensor]:
        """Return a detached CPU artifact suitable for ``torch.save``."""

        return {
            field: value.detach().cpu()
            for field, value in self.__dict__.items()
        }


def upstream_score_standard_deviation(
    time: Tensor,
    beta_schedule: BetaSchedule,
) -> Tensor:
    r"""Return upstream's ``sqrt(1-exp(-tau(t)))`` score scale."""

    tau = time if beta_schedule is None else beta_schedule.rescale_t(time)
    return torch.sqrt(1.0 - torch.exp(-tau))


def _validate_inputs(
    terminal_points: Tensor,
    standard_noise: Tensor,
    *,
    n_steps: int,
) -> None:
    if terminal_points.ndim != 2 or terminal_points.shape[1] != 3:
        raise ValueError("terminal_points must have shape [batch, 3]")
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    expected_shape = (n_steps, terminal_points.shape[0], 3)
    if tuple(standard_noise.shape) != expected_shape:
        raise ValueError(
            f"standard_noise must have shape {expected_shape}, "
            f"got {tuple(standard_noise.shape)}"
        )
    if standard_noise.device != terminal_points.device:
        raise ValueError("terminal_points and standard_noise must share a device")
    if standard_noise.dtype != terminal_points.dtype:
        raise ValueError("terminal_points and standard_noise must share a dtype")


def _stack_norms(values: list[Tensor]) -> Tensor:
    return torch.stack(values, dim=0)


def trace_s2_reverse_grw_current_style(
    terminal_points: Tensor,
    score_fn: Callable[[Tensor, Tensor], Tensor],
    *,
    terminal_time: float,
    n_steps: int,
    standard_noise: Tensor,
    minimum_forward_time: float = 1e-3,
    beta_schedule: BetaSchedule = None,
) -> S2ReverseSamplerDiagnostics:
    """Trace a copy of the current sampler without changing its implementation.

    Tests require this diagnostic copy's final samples to match the production
    :func:`s2_reverse_grw`.  This function is not used by training or ordinary
    reverse sampling.
    """

    _validate_inputs(terminal_points, standard_noise, n_steps=n_steps)
    points = terminal_points / torch.linalg.vector_norm(
        terminal_points, dim=1, keepdim=True
    )
    trajectory = [points]
    times = []
    network_norms = []
    score_norms = []
    projected_norms = []
    beta_norms = []
    beta_projected_norms = []
    drift_norms = []
    noise_norms = []
    score_stds = []

    legacy = beta_schedule is None or isinstance(
        beta_schedule, LegacyUnitBetaSchedule
    )
    if legacy:
        dt = terminal_time / n_steps
    else:
        if terminal_time <= beta_schedule.t0 or terminal_time > beta_schedule.tf:
            raise ValueError("terminal_time is outside the beta schedule")
        dt = (terminal_time - beta_schedule.t0) / n_steps

    for step in range(n_steps):
        if legacy:
            physical_time = terminal_time - step * dt
            delta_tau = dt
        else:
            physical_time = terminal_time - step * dt
            next_physical_time = terminal_time - (step + 1) * dt
            delta_tau = beta_schedule.interval_brownian_time(
                next_physical_time,
                physical_time,
            )
        forward_time = max(physical_time, minimum_forward_time)
        time_batch = torch.full(
            (points.shape[0],),
            forward_time,
            dtype=points.dtype,
            device=points.device,
        )
        network_output = score_fn(time_batch, points)
        projector = _batched_s2_projector(points)
        projected_score = torch.einsum("bij,bj->bi", projector, network_output)
        projected_noise = torch.einsum(
            "bij,bj->bi", projector, standard_noise[step]
        )
        beta = (
            torch.ones_like(time_batch)
            if beta_schedule is None
            else beta_schedule.beta_t(time_batch)
        )
        drift_increment = float(delta_tau) * projected_score
        noise_increment = math.sqrt(float(delta_tau)) * projected_noise
        tangent_increment = drift_increment + noise_increment
        points = torch.stack(
            [
                s2_exp(point, increment)
                for point, increment in zip(points, tangent_increment)
            ]
        )

        trajectory.append(points)
        times.append(time_batch[0])
        network_norms.append(torch.linalg.vector_norm(network_output, dim=1))
        score_norms.append(torch.linalg.vector_norm(network_output, dim=1))
        projected_norms.append(torch.linalg.vector_norm(projected_score, dim=1))
        beta_norms.append(
            torch.linalg.vector_norm(beta[:, None] * network_output, dim=1)
        )
        beta_projected_norms.append(
            torch.linalg.vector_norm(beta[:, None] * projected_score, dim=1)
        )
        drift_norms.append(torch.linalg.vector_norm(drift_increment, dim=1))
        noise_norms.append(torch.linalg.vector_norm(noise_increment, dim=1))
        score_stds.append(torch.ones_like(time_batch))

    return S2ReverseSamplerDiagnostics(
        final_samples=points,
        trajectory=torch.stack(trajectory, dim=0),
        time_grid=torch.stack(times),
        network_output_norm=_stack_norms(network_norms),
        score_norm=_stack_norms(score_norms),
        projected_score_norm=_stack_norms(projected_norms),
        beta_score_norm=_stack_norms(beta_norms),
        beta_projected_score_norm=_stack_norms(beta_projected_norms),
        drift_increment_norm=_stack_norms(drift_norms),
        noise_increment_norm=_stack_norms(noise_norms),
        score_std=_stack_norms(score_stds),
    )


def s2_reverse_grw_upstream_style(
    terminal_points: Tensor,
    network_output_fn: Callable[[Tensor, Tensor], Tensor],
    *,
    standard_noise: Tensor,
    beta_schedule: BetaSchedule,
    terminal_time: float = 1.0,
    epsilon: float = UPSTREAM_REVERSE_EPSILON,
    n_steps: int = UPSTREAM_REVERSE_STEPS,
    divide_network_output_by_std: bool = True,
) -> S2ReverseSamplerDiagnostics:
    r"""Run the upstream Earthquake reverse-SDE/GRW discretisation.

    Compatibility details intentionally reproduced here are:

    * ``N=100`` and physical time from ``1`` to ``epsilon=0.001``;
    * ``N`` endpoint-inclusive score-evaluation times;
    * signed ``dt=(epsilon-terminal_time)/N``;
    * ``score=network_output/sqrt(1-exp(-tau(t)))``;
    * reverse drift ``-beta(t)*score`` integrated with the negative ``dt``;
    * tangent noise scale ``sqrt(beta(t)*abs(dt))``.

    The caller supplies ambient standard normal noise.  It is projected to the
    tangent space, matching the distribution produced by upstream geomstats'
    ``random_normal_tangent`` on the embedded unit sphere.
    """

    _validate_inputs(terminal_points, standard_noise, n_steps=n_steps)
    if not 0.0 < epsilon < terminal_time:
        raise ValueError("epsilon must satisfy 0 < epsilon < terminal_time")
    points = terminal_points / torch.linalg.vector_norm(
        terminal_points, dim=1, keepdim=True
    )
    time_grid = torch.linspace(
        terminal_time,
        epsilon,
        steps=n_steps,
        dtype=points.dtype,
        device=points.device,
    )
    signed_dt = (epsilon - terminal_time) / n_steps
    absolute_dt = abs(signed_dt)
    trajectory = [points]
    network_norms = []
    score_norms = []
    projected_norms = []
    beta_norms = []
    beta_projected_norms = []
    drift_norms = []
    noise_norms = []
    score_stds = []

    for step in range(n_steps):
        time_batch = time_grid[step].expand(points.shape[0])
        network_output = network_output_fn(time_batch, points)
        score_std = upstream_score_standard_deviation(time_batch, beta_schedule)
        score = (
            network_output / score_std[:, None]
            if divide_network_output_by_std
            else network_output
        )
        projector = _batched_s2_projector(points)
        projected_score = torch.einsum("bij,bj->bi", projector, score)
        projected_noise = torch.einsum(
            "bij,bj->bi", projector, standard_noise[step]
        )
        beta = (
            torch.ones_like(time_batch)
            if beta_schedule is None
            else beta_schedule.beta_t(time_batch)
        )
        reverse_drift = -beta[:, None] * projected_score
        drift_increment = signed_dt * reverse_drift
        noise_increment = torch.sqrt(beta * absolute_dt)[:, None] * projected_noise
        tangent_increment = drift_increment + noise_increment
        points = torch.stack(
            [
                s2_exp(point, increment)
                for point, increment in zip(points, tangent_increment)
            ]
        )

        trajectory.append(points)
        network_norms.append(torch.linalg.vector_norm(network_output, dim=1))
        score_norms.append(torch.linalg.vector_norm(score, dim=1))
        projected_norms.append(torch.linalg.vector_norm(projected_score, dim=1))
        beta_norms.append(
            torch.linalg.vector_norm(beta[:, None] * score, dim=1)
        )
        beta_projected_norms.append(
            torch.linalg.vector_norm(beta[:, None] * projected_score, dim=1)
        )
        drift_norms.append(torch.linalg.vector_norm(drift_increment, dim=1))
        noise_norms.append(torch.linalg.vector_norm(noise_increment, dim=1))
        score_stds.append(score_std)

    return S2ReverseSamplerDiagnostics(
        final_samples=points,
        trajectory=torch.stack(trajectory, dim=0),
        time_grid=time_grid,
        network_output_norm=_stack_norms(network_norms),
        score_norm=_stack_norms(score_norms),
        projected_score_norm=_stack_norms(projected_norms),
        beta_score_norm=_stack_norms(beta_norms),
        beta_projected_score_norm=_stack_norms(beta_projected_norms),
        drift_increment_norm=_stack_norms(drift_norms),
        noise_increment_norm=_stack_norms(noise_norms),
        score_std=_stack_norms(score_stds),
    )
