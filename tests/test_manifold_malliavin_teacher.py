"""Tests for the generic discrete Malliavin--Skorokhod backend.

These tests are intended to run on the GPU server.  The exact Skorokhod
divergence differentiates through a Jacobian and can be slow on a laptop.
"""

import math

import torch

from scoremodel_ext.malliavin.models import MirafzaliSkorokhodNet
from scoremodel_ext.manifold.malliavin_teacher import (
    discrete_malliavin_skorokhod_teacher,
    tangent_malliavin_skorokhod_teacher,
)
from scoremodel_ext.manifold.s2_malliavin import (
    S2SkorokhodScoreModel,
    s2_discrete_malliavin_teacher,
    s2_grw_endpoint,
    s2_heat_kernel_score,
    s2_projector,
    s2_reconstruct_score_vector,
    s2_reverse_grw,
    s2_tangent_basis,
    s2_tangent_malliavin_teacher,
    s2_varadhan_score,
)


DTYPE = torch.float64


def test_mirafzali_network_supports_distinct_input_and_teacher_dimensions():
    network = MirafzaliSkorokhodNet(
        x_dim=5,
        out_dim=2,
        hidden=16,
        n_blocks=1,
        num_frequencies=4,
    )
    output = network(torch.rand(3), torch.rand(3, 5))
    assert output.shape == (3, 2)


def test_euclidean_additive_noise_recovers_exact_path_weight():
    """For X_t=x0+sqrt(t)Z the score weight is exactly -Z/sqrt(t)."""

    x0 = torch.tensor([0.4, -0.7], dtype=DTYPE)
    z = torch.tensor([0.3, -1.2], dtype=DTYPE)
    time = 0.6
    identity = torch.eye(2, dtype=DTYPE)

    teacher = discrete_malliavin_skorokhod_teacher(
        lambda noise: x0 + math.sqrt(time) * noise,
        z,
        lambda endpoint: identity,
        field_divergence_fn=lambda endpoint: torch.zeros(2, dtype=DTYPE),
        covariance_regularization=1e-12,
    )

    expected = -z / math.sqrt(time)
    torch.testing.assert_close(teacher.skorokhod, -expected, rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(teacher.score_weight, expected, rtol=1e-8, atol=1e-8)


def test_s2_grw_endpoint_remains_on_sphere():
    x0 = torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE)
    noise = torch.tensor(
        [[0.2, -0.4, 0.7], [-0.1, 0.8, 0.3], [0.5, 0.2, -0.6]],
        dtype=DTYPE,
    )
    endpoint = s2_grw_endpoint(x0, noise, terminal_time=0.2)
    torch.testing.assert_close(
        torch.linalg.vector_norm(endpoint),
        torch.ones((), dtype=DTYPE),
        rtol=1e-10,
        atol=1e-10,
    )


def test_tangent_malliavin_teacher_matches_simple_additive_noise():
    x0 = torch.tensor([0.4, -0.7], dtype=DTYPE)
    z = torch.tensor([0.3, -1.2], dtype=DTYPE)
    time = 0.6
    tangent_basis = torch.eye(2, dtype=DTYPE)

    teacher = tangent_malliavin_skorokhod_teacher(
        lambda noise: x0 + math.sqrt(time) * noise,
        z,
        lambda endpoint: tangent_basis,
        lambda endpoint: torch.tensor([1.0, 0.0], dtype=DTYPE),
        field_divergence_fn=lambda endpoint: torch.zeros((), dtype=DTYPE),
        covariance_regularization=1e-12,
    )

    expected = -z / math.sqrt(time)
    torch.testing.assert_close(teacher.skorokhod, expected, rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(teacher.covering, z / math.sqrt(time), rtol=1e-8, atol=1e-8)


def test_s2_tangent_basis_and_reconstruction_are_tangent():
    endpoint = torch.tensor([0.2, -0.3, 0.9327379053], dtype=DTYPE)
    endpoint = endpoint / torch.linalg.vector_norm(endpoint)
    basis = s2_tangent_basis(endpoint)
    directional_scores = torch.tensor([0.4, -0.1, 0.3], dtype=DTYPE)

    reconstructed = s2_reconstruct_score_vector(directional_scores, endpoint)

    assert basis.shape == (3, 2)
    assert abs(float(torch.dot(reconstructed, endpoint))) < 1e-10
    torch.testing.assert_close(
        reconstructed,
        basis @ directional_scores[:2],
        rtol=1e-10,
        atol=1e-10,
    )


def test_s2_tangent_teacher_is_tangent_and_covariance_has_two_tangent_modes():
    x0 = torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE)
    noise = torch.tensor(
        [[0.2, -0.4, 0.7], [-0.1, 0.8, 0.3]],
        dtype=DTYPE,
    )
    teacher = s2_tangent_malliavin_teacher(
        x0,
        noise,
        terminal_time=0.2,
        field_index=0,
        covariance_regularization=1e-8,
    )

    tangent_residual = torch.dot(teacher.endpoint, teacher.score_weight)
    assert abs(float(tangent_residual)) < 1e-7
    assert teacher.covariance_eigenvalues.shape == (2,)
    assert teacher.covariance_eigenvalues[0] > 1e-6
    assert teacher.covariance_eigenvalues[1] > 1e-6


def test_s2_teacher_is_tangent_and_covariance_has_two_tangent_modes():
    x0 = torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE)
    noise = torch.tensor(
        [[0.2, -0.4, 0.7], [-0.1, 0.8, 0.3]],
        dtype=DTYPE,
    )
    teacher = s2_discrete_malliavin_teacher(
        x0,
        noise,
        terminal_time=0.2,
        covariance_regularization=1e-8,
    )

    tangent_residual = torch.dot(teacher.endpoint, teacher.score_weight)
    assert abs(float(tangent_residual)) < 1e-7
    assert teacher.covariance_eigenvalues.shape == (3,)
    assert teacher.covariance_eigenvalues[-1] > 1e-6
    assert teacher.covariance_eigenvalues[-2] > 1e-6
    assert abs(float(teacher.covariance_eigenvalues[0])) < 1e-7


def test_s2_heat_kernel_and_varadhan_scores_are_tangent():
    x0 = torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE)
    endpoint = torch.tensor([0.2, -0.3, 0.9327379053], dtype=DTYPE)
    endpoint = endpoint / torch.linalg.vector_norm(endpoint)

    exact = s2_heat_kernel_score(x0, endpoint, terminal_time=0.3, n_terms=60)
    asymptotic = s2_varadhan_score(x0, endpoint, terminal_time=0.3)
    projector = s2_projector(endpoint)

    torch.testing.assert_close(projector @ exact, exact, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(
        projector @ asymptotic, asymptotic, rtol=1e-9, atol=1e-9
    )


def test_s2_heat_kernel_score_points_towards_initial_point():
    x0 = torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE)
    endpoint = torch.tensor([0.4, 0.0, math.sqrt(0.84)], dtype=DTYPE)
    score = s2_heat_kernel_score(x0, endpoint, terminal_time=0.4, n_terms=60)
    direction_to_x0 = s2_projector(endpoint) @ x0
    assert torch.dot(score, direction_to_x0) > 0


def test_s2_skorokhod_wrapper_projects_network_output():
    class ConstantDelta(torch.nn.Module):
        def forward(self, t, x):
            return torch.tensor([1.0, -2.0, 0.5], dtype=x.dtype).expand_as(x)

    model = S2SkorokhodScoreModel(ConstantDelta())
    points = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=DTYPE
    )
    score = model(torch.tensor([0.2, 0.2], dtype=DTYPE), points)
    assert torch.max(torch.abs((score * points).sum(dim=1))) < 1e-12


def test_reverse_grw_keeps_all_samples_on_sphere():
    terminal = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=DTYPE
    )
    noise = torch.zeros(3, 2, 3, dtype=DTYPE)
    output = s2_reverse_grw(
        terminal,
        lambda t, x: torch.zeros_like(x),
        terminal_time=0.3,
        n_steps=3,
        standard_noise=noise,
    )
    torch.testing.assert_close(output, terminal)
