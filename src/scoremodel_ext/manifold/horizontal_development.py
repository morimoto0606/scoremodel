"""Horizontal development without the Euclidean-score-lift assumption.

This module only implements geometry and a differentiable Stratonovich
integrator.  Probabilistic scores are supplied by the generic endpoint-map
Malliavin backend, not by identifying the frame-bundle density with the
density of a Euclidean endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import torch

from .malliavin_teacher import (
    DiscreteMalliavinTeacher,
    discrete_malliavin_skorokhod_teacher,
)


Tensor = torch.Tensor


class CoordinateGeometry(Protocol):
    """Local-coordinate geometry required by horizontal development."""

    def metric(self, point: Tensor) -> Tensor:
        """Return ``g_ij(point)`` with shape ``[d,d]``."""

    def christoffel(self, point: Tensor) -> Tensor:
        """Return ``Gamma^k_ij(point)`` with shape ``[d,d,d]``."""

    def retract(self, point: Tensor, coordinate_increment: Tensor) -> Tensor:
        """Move to a new chart point; ``point + increment`` is allowed."""


@dataclass
class HorizontalFrameState:
    """A local-coordinate point and a frame ``R^d -> T_x M``."""

    point: Tensor
    frame: Tensor

    def flatten(self) -> Tensor:
        return torch.cat((self.point.reshape(-1), self.frame.reshape(-1)))


def unpack_horizontal_state(flat_state: Tensor, dimension: int) -> HorizontalFrameState:
    """Inverse operation to :meth:`HorizontalFrameState.flatten`."""

    expected = dimension + dimension * dimension
    if flat_state.numel() != expected:
        raise ValueError(f"flat state has {flat_state.numel()} entries, expected {expected}")
    point = flat_state[:dimension]
    frame = flat_state[dimension:].reshape(dimension, dimension)
    return HorizontalFrameState(point=point, frame=frame)


def horizontal_lift(
    geometry: CoordinateGeometry,
    state: HorizontalFrameState,
    frame_coordinates: Tensor,
) -> HorizontalFrameState:
    r"""Apply the canonical horizontal lift to a vector in ``R^d``.

    In coordinates,

    .. math::

        \dot x^i=e_a^i w^a,\qquad
        \dot e_b^k=-\Gamma^k_{ij}(x)\dot x^i e_b^j.
    """

    dimension = state.point.numel()
    if state.frame.shape != (dimension, dimension):
        raise ValueError("frame must have shape [dimension, dimension]")
    w = frame_coordinates.reshape(dimension)
    base_velocity = state.frame @ w
    gamma = geometry.christoffel(state.point)
    if gamma.shape != (dimension, dimension, dimension):
        raise ValueError("christoffel must have shape [dimension, dimension, dimension]")
    frame_velocity = -torch.einsum(
        "kij,i,jb->kb",
        gamma,
        base_velocity,
        state.frame,
    )
    return HorizontalFrameState(base_velocity, frame_velocity)


def metric_orthonormalize(
    geometry: CoordinateGeometry,
    point: Tensor,
    frame: Tensor,
    *,
    jitter: float = 1e-10,
) -> Tensor:
    """Differentiably enforce ``frame.T @ g(point) @ frame = I``."""

    gram = frame.transpose(0, 1) @ geometry.metric(point) @ frame
    gram = 0.5 * (gram + gram.transpose(0, 1))
    eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    lower = torch.linalg.cholesky(gram + jitter * eye)
    inverse_lower_transpose = torch.linalg.solve_triangular(
        lower.transpose(0, 1),
        eye,
        upper=True,
    )
    return frame @ inverse_lower_transpose


def horizontal_heun_step(
    geometry: CoordinateGeometry,
    state: HorizontalFrameState,
    driver_increment: Tensor,
    *,
    orthonormalize: bool = True,
    frame_jitter: float = 1e-10,
) -> HorizontalFrameState:
    """One explicit Heun step for ``dU=H(U) o dE``.

    ``driver_increment`` may contain both the horizontal drift times ``dt``
    and the stochastic increment.  Re-evaluating the horizontal vector field
    at the predictor implements the Stratonovich correction without converting
    the SDE to an Itô coordinate equation.
    """

    first = horizontal_lift(geometry, state, driver_increment)
    predictor = HorizontalFrameState(
        point=geometry.retract(state.point, first.point),
        frame=state.frame + first.frame,
    )
    second = horizontal_lift(geometry, predictor, driver_increment)
    point_increment = 0.5 * (first.point + second.point)
    frame_increment = 0.5 * (first.frame + second.frame)
    point = geometry.retract(state.point, point_increment)
    frame = state.frame + frame_increment
    if orthonormalize:
        frame = metric_orthonormalize(
            geometry,
            point,
            frame,
            jitter=frame_jitter,
        )
    return HorizontalFrameState(point=point, frame=frame)


def horizontal_development_endpoint(
    geometry: CoordinateGeometry,
    initial_state: HorizontalFrameState,
    driver_increments: Tensor,
    *,
    return_frame_bundle: bool,
    orthonormalize_every_step: bool = True,
    frame_jitter: float = 1e-10,
) -> Tensor:
    """Develop a Euclidean path and return either ``X_t`` or ``U_t``."""

    dimension = initial_state.point.numel()
    if driver_increments.ndim != 2 or driver_increments.shape[1] != dimension:
        raise ValueError("driver_increments must have shape [n_steps, dimension]")
    state = initial_state
    for increment in driver_increments:
        state = horizontal_heun_step(
            geometry,
            state,
            increment,
            orthonormalize=orthonormalize_every_step,
            frame_jitter=frame_jitter,
        )
    return state.flatten() if return_frame_bundle else state.point


def horizontal_malliavin_teacher(
    geometry: CoordinateGeometry,
    initial_state: HorizontalFrameState,
    standard_noise: Tensor,
    driver_from_standard_noise: Callable[[Tensor], Tensor],
    target_fields_fn: Callable[[Tensor], Tensor],
    field_divergence_fn: Callable[[Tensor], Tensor],
    *,
    return_frame_bundle: bool,
    covariance_regularization: float = 1e-6,
    vectorize_jacobian: bool = True,
) -> DiscreteMalliavinTeacher:
    """Apply the shared Malliavin backend to horizontal development.

    ``driver_from_standard_noise`` is responsible for the time step, noise
    schedule and any horizontal drift.  No Euclidean endpoint density or
    Euclidean score is used.
    """

    def endpoint_fn(noise: Tensor) -> Tensor:
        driver = driver_from_standard_noise(noise)
        return horizontal_development_endpoint(
            geometry,
            initial_state,
            driver,
            return_frame_bundle=return_frame_bundle,
        )

    return discrete_malliavin_skorokhod_teacher(
        endpoint_fn,
        standard_noise,
        target_fields_fn,
        field_divergence_fn=field_divergence_fn,
        covariance_regularization=covariance_regularization,
        vectorize_jacobian=vectorize_jacobian,
    )
