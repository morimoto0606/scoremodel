"""Geometry-only tests for Park-style horizontal development.

No Euclidean-score-lift identity is used in these tests.
"""

import torch

from scoremodel_ext.manifold.horizontal_development import (
    HorizontalFrameState,
    horizontal_development_endpoint,
    horizontal_heun_step,
    horizontal_lift,
    metric_orthonormalize,
)


class FlatGeometry:
    def metric(self, point):
        return torch.eye(point.numel(), dtype=point.dtype, device=point.device)

    def christoffel(self, point):
        d = point.numel()
        return torch.zeros(d, d, d, dtype=point.dtype, device=point.device)

    def retract(self, point, coordinate_increment):
        return point + coordinate_increment


class ConformalPlaneGeometry:
    """Metric exp(2*x0) I with analytically known Christoffel symbols."""

    def metric(self, point):
        return torch.exp(2.0 * point[0]) * torch.eye(2, dtype=point.dtype)

    def christoffel(self, point):
        gamma = torch.zeros(2, 2, 2, dtype=point.dtype)
        # For phi=x0: Gamma^k_ij = delta^k_j phi_i + delta^k_i phi_j
        #                         - delta_ij phi^k.
        gamma[0, 0, 0] = 1.0
        gamma[0, 1, 1] = -1.0
        gamma[1, 0, 1] = 1.0
        gamma[1, 1, 0] = 1.0
        return gamma

    def retract(self, point, coordinate_increment):
        return point + coordinate_increment


def test_flat_horizontal_lift_keeps_frame_constant():
    geometry = FlatGeometry()
    state = HorizontalFrameState(
        point=torch.zeros(2, dtype=torch.float64),
        frame=torch.eye(2, dtype=torch.float64),
    )
    velocity = horizontal_lift(
        geometry,
        state,
        torch.tensor([0.2, -0.4], dtype=torch.float64),
    )
    torch.testing.assert_close(velocity.point, torch.tensor([0.2, -0.4], dtype=torch.float64))
    torch.testing.assert_close(velocity.frame, torch.zeros(2, 2, dtype=torch.float64))


def test_flat_heun_development_equals_sum_of_driver_increments():
    geometry = FlatGeometry()
    initial = HorizontalFrameState(
        point=torch.tensor([0.3, -0.2], dtype=torch.float64),
        frame=torch.eye(2, dtype=torch.float64),
    )
    increments = torch.tensor(
        [[0.1, 0.2], [-0.4, 0.3], [0.2, -0.1]], dtype=torch.float64
    )
    endpoint = horizontal_development_endpoint(
        geometry,
        initial,
        increments,
        return_frame_bundle=False,
    )
    torch.testing.assert_close(endpoint, initial.point + increments.sum(dim=0))


def test_metric_orthonormalization_enforces_constraint():
    geometry = ConformalPlaneGeometry()
    point = torch.tensor([0.4, -0.1], dtype=torch.float64)
    frame = torch.tensor([[1.2, 0.3], [-0.2, 0.8]], dtype=torch.float64)
    orthonormal = metric_orthonormalize(geometry, point, frame)
    gram = orthonormal.T @ geometry.metric(point) @ orthonormal
    torch.testing.assert_close(gram, torch.eye(2, dtype=torch.float64), rtol=1e-9, atol=1e-9)


def test_horizontal_heun_preserves_metric_frame_after_reorthonormalization():
    geometry = ConformalPlaneGeometry()
    point = torch.tensor([0.1, -0.2], dtype=torch.float64)
    frame = metric_orthonormalize(geometry, point, torch.eye(2, dtype=torch.float64))
    state = HorizontalFrameState(point, frame)
    next_state = horizontal_heun_step(
        geometry,
        state,
        torch.tensor([0.02, -0.01], dtype=torch.float64),
    )
    gram = next_state.frame.T @ geometry.metric(next_state.point) @ next_state.frame
    torch.testing.assert_close(gram, torch.eye(2, dtype=torch.float64), rtol=1e-8, atol=1e-8)
