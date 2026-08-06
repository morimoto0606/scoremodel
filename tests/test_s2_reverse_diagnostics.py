import math

import torch

from scoremodel_ext.manifold.beta_schedule import LinearBetaSchedule
from scoremodel_ext.manifold.s2_malliavin import (
    s2_exp,
    s2_projector,
    s2_reverse_grw,
)
from scoremodel_ext.manifold.s2_reverse_diagnostics import (
    UPSTREAM_REVERSE_EPSILON,
    UPSTREAM_REVERSE_STEPS,
    s2_reverse_grw_upstream_style,
    trace_s2_reverse_grw_current_style,
    upstream_score_standard_deviation,
)


DTYPE = torch.float64


def test_upstream_score_standard_deviation_uses_rescaled_brownian_time():
    schedule = LinearBetaSchedule(
        beta_0=0.001,
        beta_f=5.0,
        t0=0.0,
        tf=1.0,
    )
    times = torch.tensor([0.001, 0.1, 0.5, 1.0], dtype=DTYPE)
    expected = torch.sqrt(1.0 - torch.exp(-schedule.rescale_t(times)))
    actual = upstream_score_standard_deviation(times, schedule)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_upstream_style_defaults_match_repository_earthquake_settings():
    assert UPSTREAM_REVERSE_STEPS == 100
    assert UPSTREAM_REVERSE_EPSILON == 0.001


def test_upstream_style_applies_std_parameterisation_beta_and_signed_dt():
    schedule = LinearBetaSchedule(
        beta_0=2.0,
        beta_f=2.0,
        t0=0.0,
        tf=1.0,
    )
    terminal = torch.tensor([[0.0, 0.0, 1.0]], dtype=DTYPE)
    raw_network_output = torch.tensor([[0.2, -0.3, 0.7]], dtype=DTYPE)
    noise = torch.tensor([[[0.4, 0.1, -0.2]]], dtype=DTYPE)
    epsilon = 0.25

    def network_output_fn(t, x):
        return raw_network_output.expand_as(x)

    result = s2_reverse_grw_upstream_style(
        terminal,
        network_output_fn,
        standard_noise=noise,
        beta_schedule=schedule,
        terminal_time=1.0,
        epsilon=epsilon,
        n_steps=1,
    )

    tau = schedule.rescale_t(torch.tensor(1.0, dtype=DTYPE))
    std = torch.sqrt(1.0 - torch.exp(-tau))
    projector = s2_projector(terminal[0])
    score = projector @ (raw_network_output[0] / std)
    tangent_noise = projector @ noise[0, 0]
    absolute_dt = 1.0 - epsilon
    expected_increment = (
        absolute_dt * 2.0 * score
        + math.sqrt(2.0 * absolute_dt) * tangent_noise
    )
    expected = s2_exp(terminal[0], expected_increment).reshape(1, 3)

    torch.testing.assert_close(result.final_samples, expected, rtol=0, atol=1e-14)
    torch.testing.assert_close(
        result.score_std,
        std.reshape(1, 1),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        result.beta_projected_score_norm[0, 0],
        2.0 * torch.linalg.vector_norm(score),
        rtol=0,
        atol=1e-14,
    )
    torch.testing.assert_close(
        result.beta_score_norm[0, 0],
        2.0 * torch.linalg.vector_norm(raw_network_output[0] / std),
        rtol=0,
        atol=1e-14,
    )
    assert tuple(result.trajectory.shape) == (2, 1, 3)


def test_current_style_diagnostic_matches_production_sampler_float64():
    schedule = LinearBetaSchedule(
        beta_0=0.001,
        beta_f=5.0,
        t0=0.0,
        tf=1.0,
    )
    terminal = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        dtype=DTYPE,
    )
    noise = torch.tensor(
        [
            [[0.2, -0.1, 0.3], [-0.2, 0.4, 0.1]],
            [[-0.4, 0.5, 0.1], [0.3, -0.1, 0.2]],
        ],
        dtype=DTYPE,
    )

    def score_fn(t, x):
        return torch.stack((0.2 * t, -0.1 * t, 0.3 * t), dim=1)

    production = s2_reverse_grw(
        terminal,
        score_fn,
        terminal_time=1.0,
        n_steps=2,
        standard_noise=noise,
        minimum_forward_time=0.001,
        beta_schedule=schedule,
    )
    diagnostic = trace_s2_reverse_grw_current_style(
        terminal,
        score_fn,
        terminal_time=1.0,
        n_steps=2,
        standard_noise=noise,
        minimum_forward_time=0.001,
        beta_schedule=schedule,
    )

    torch.testing.assert_close(
        diagnostic.final_samples,
        production,
        rtol=0,
        atol=0,
    )
    assert tuple(diagnostic.trajectory.shape) == (3, 2, 3)
    torch.testing.assert_close(
        diagnostic.time_grid,
        torch.tensor([1.0, 0.5], dtype=DTYPE),
        rtol=0,
        atol=0,
    )


def test_upstream_style_uses_endpoint_inclusive_score_time_grid():
    terminal = torch.tensor([[0.0, 0.0, 1.0]], dtype=DTYPE)
    noise = torch.zeros(4, 1, 3, dtype=DTYPE)

    def zero_network(t, x):
        return torch.zeros_like(x)

    result = s2_reverse_grw_upstream_style(
        terminal,
        zero_network,
        standard_noise=noise,
        beta_schedule=None,
        terminal_time=1.0,
        epsilon=0.001,
        n_steps=4,
    )
    expected = torch.linspace(1.0, 0.001, steps=4, dtype=DTYPE)
    torch.testing.assert_close(result.time_grid, expected, rtol=0, atol=0)
