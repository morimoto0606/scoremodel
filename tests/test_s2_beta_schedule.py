from types import SimpleNamespace

import pytest
import torch

from scripts import experiment_earthquake_teacher_compare_smoke as runner
from scoremodel_ext.manifold.beta_schedule import (
    LegacyUnitBetaSchedule,
    LinearBetaSchedule,
)
from scoremodel_ext.manifold.s2_malliavin import (
    s2_discrete_malliavin_teacher,
    s2_exp,
    s2_projector,
    s2_reverse_grw,
)


DTYPE = torch.float64
MALLIAVIN_DETAIL_KEYS = (
    "endpoint",
    "endpoint_jacobian",
    "covariance",
    "covering",
    "divergence_term",
    "skorokhod",
    "score_weight",
)


def test_linear_beta_schedule_matches_analytic_formula_float64():
    schedule = LinearBetaSchedule(beta_0=0.3, beta_f=2.1, t0=-0.5, tf=1.5)
    times = torch.tensor([-0.5, -0.1, 0.4, 1.5], dtype=DTYPE)
    u = (times + 0.5) / 2.0
    expected_beta = 0.3 + u * (2.1 - 0.3)
    expected_tau = 0.3 * u + 0.5 * (2.1 - 0.3) * u**2

    torch.testing.assert_close(schedule.beta_t(times), expected_beta, rtol=0, atol=1e-14)
    torch.testing.assert_close(
        schedule.rescale_t(times), expected_tau, rtol=0, atol=1e-14
    )
    torch.testing.assert_close(
        schedule.interval_brownian_time(times[:-1], times[1:]),
        expected_tau[1:] - expected_tau[:-1],
        rtol=0,
        atol=1e-14,
    )
    assert schedule.beta_t(0.4) == pytest.approx(1.11)


def test_linear_beta_schedule_is_vmap_safe():
    schedule = LinearBetaSchedule(beta_0=0.001, beta_f=5.0, t0=0.0, tf=1.0)
    times = torch.linspace(0.0, 1.0, 5, dtype=DTYPE)
    vmapped = torch.func.vmap(schedule.rescale_t)(times)
    torch.testing.assert_close(vmapped, schedule.rescale_t(times), rtol=0, atol=0)


def test_de_bortoli_upstream_linear_schedule_conformance():
    schedule = LinearBetaSchedule(beta_0=0.001, beta_f=5.0, t0=0.0, tf=1.0)
    times = torch.tensor([0.0, 0.001, 0.05, 0.3, 0.75, 1.0], dtype=DTYPE)
    normalized_time = times
    upstream_beta = 0.001 + normalized_time * (5.0 - 0.001)
    upstream_rescaled = (
        0.001 * normalized_time
        + 0.5 * normalized_time**2 * (5.0 - 0.001)
    )
    torch.testing.assert_close(
        schedule.beta_t(times), upstream_beta, rtol=0, atol=1e-14
    )
    torch.testing.assert_close(
        schedule.rescale_t(times), upstream_rescaled, rtol=0, atol=1e-14
    )
    restored = runner.beta_schedule_from_run_config(
        {
            "beta_schedule": "linear",
            "beta_0": 0.001,
            "beta_f": 5.0,
            "beta_t0": 0.0,
            "beta_tf": 1.0,
        }
    )
    assert restored == schedule


def test_legacy_schedule_and_missing_run_config_use_legacy_path():
    legacy_schedule = LegacyUnitBetaSchedule()
    legacy_times = torch.tensor([0.05, 0.2, 0.3], dtype=DTYPE)
    torch.testing.assert_close(
        legacy_schedule.beta_t(legacy_times),
        torch.ones_like(legacy_times),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        legacy_schedule.rescale_t(legacy_times), legacy_times, rtol=0, atol=0
    )
    torch.testing.assert_close(
        legacy_schedule.interval_brownian_time(
            legacy_times[:-1], legacy_times[1:]
        ),
        legacy_times[1:] - legacy_times[:-1],
        rtol=0,
        atol=0,
    )
    assert runner.beta_schedule_from_run_config({}) is None
    assert runner.build_beta_schedule(
        "legacy-unit",
        beta_0=0.001,
        beta_f=5.0,
        beta_t0=0.0,
        beta_tf=1.0,
    ) is None

    terminal = torch.tensor([[0.0, 0.0, 1.0]], dtype=DTYPE)
    noise = torch.tensor(
        [[[0.2, -0.1, 0.3]], [[-0.4, 0.5, 0.1]]], dtype=DTYPE
    )

    def score_fn(t, x):
        return torch.stack((t, -t, torch.zeros_like(t)), dim=1)

    omitted = s2_reverse_grw(
        terminal,
        score_fn,
        terminal_time=0.2,
        n_steps=2,
        standard_noise=noise,
        minimum_forward_time=0.01,
    )
    explicit_legacy = s2_reverse_grw(
        terminal,
        score_fn,
        terminal_time=0.2,
        n_steps=2,
        standard_noise=noise,
        minimum_forward_time=0.01,
        beta_schedule=LegacyUnitBetaSchedule(),
    )
    torch.testing.assert_close(omitted, explicit_legacy, rtol=0, atol=0)


@pytest.mark.parametrize("teacher", ["heat", "varadhan", "malliavin"])
def test_all_teachers_receive_common_rescaled_time_and_store_physical_time(
    monkeypatch,
    teacher,
):
    schedule = LinearBetaSchedule(beta_0=0.001, beta_f=5.0, t0=0.0, tf=1.0)
    physical_time = torch.tensor([0.3], dtype=DTYPE)
    expected_tau = float(schedule.rescale_t(physical_time[0]))
    observed_times = []
    endpoint = torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE)
    score = torch.tensor([0.2, -0.1, 0.0], dtype=DTYPE)

    def fake_grw(initial_point, noise, terminal_time):
        observed_times.append(terminal_time)
        return endpoint

    def fake_heat(initial_point, endpoint_value, terminal_time, **kwargs):
        observed_times.append(terminal_time)
        return score

    def fake_varadhan(initial_point, endpoint_value, terminal_time):
        observed_times.append(terminal_time)
        return score

    def fake_malliavin(initial_point, noise, terminal_time, **kwargs):
        observed_times.append(terminal_time)
        return SimpleNamespace(endpoint=endpoint, score_weight=score, skorokhod=-score)

    monkeypatch.setattr(runner, "s2_grw_endpoint", fake_grw)
    monkeypatch.setattr(runner, "s2_heat_kernel_score", fake_heat)
    monkeypatch.setattr(runner, "s2_varadhan_score", fake_varadhan)
    monkeypatch.setattr(runner, "s2_discrete_malliavin_teacher", fake_malliavin)

    dataset = runner.build_teacher_dataset(
        initial_points=torch.tensor([[0.0, 0.0, 1.0]], dtype=DTYPE),
        times=physical_time,
        noises=torch.zeros(1, 2, 3, dtype=DTYPE),
        teacher=teacher,
        covariance_regularization=1e-6,
        heat_terms=8,
        beta_schedule=schedule,
    )

    assert observed_times
    assert all(value == pytest.approx(expected_tau) for value in observed_times)
    torch.testing.assert_close(dataset["time"], physical_time, rtol=0, atol=0)


def test_linear_schedule_scalar_and_batched_malliavin_match_float64():
    schedule = LinearBetaSchedule(beta_0=0.001, beta_f=5.0, t0=0.0, tf=1.0)
    generator = torch.Generator(device="cpu").manual_seed(1729)
    initial_points = torch.randn(2, 3, generator=generator, dtype=DTYPE)
    initial_points = initial_points / torch.linalg.vector_norm(
        initial_points, dim=1, keepdim=True
    )
    noises = torch.randn(2, 2, 3, generator=generator, dtype=DTYPE)
    physical_times = torch.tensor([0.1, 0.3], dtype=DTYPE)
    brownian_times = schedule.rescale_t(physical_times)

    dataset, details, _ = runner.build_malliavin_teacher_dataset_batched(
        initial_points=initial_points,
        times=physical_times,
        noises=noises,
        batch_size=2,
        covariance_regularization=1e-6,
        beta_schedule=schedule,
    )
    torch.testing.assert_close(dataset["time"], physical_times, rtol=0, atol=0)

    for index in range(2):
        scalar = s2_discrete_malliavin_teacher(
            initial_points[index],
            noises[index],
            float(brownian_times[index]),
            covariance_regularization=1e-6,
        )
        for key in MALLIAVIN_DETAIL_KEYS:
            torch.testing.assert_close(
                details[key][index],
                getattr(scalar, key),
                rtol=0,
                atol=1e-12,
            )


def test_reverse_linear_constant_beta_uses_delta_tau_for_drift_and_noise():
    beta_value = 2.5
    terminal_time = 0.4
    schedule = LinearBetaSchedule(
        beta_0=beta_value,
        beta_f=beta_value,
        t0=0.0,
        tf=1.0,
    )
    terminal = torch.tensor([[0.0, 0.0, 1.0]], dtype=DTYPE)
    noise = torch.tensor([[[0.4, -0.2, 0.3]]], dtype=DTYPE)
    raw_score = torch.tensor([[0.1, 0.3, -0.2]], dtype=DTYPE)
    received_times = []

    def score_fn(t, x):
        received_times.append(t.clone())
        return raw_score.expand_as(x)

    actual = s2_reverse_grw(
        terminal,
        score_fn,
        terminal_time=terminal_time,
        n_steps=1,
        standard_noise=noise,
        minimum_forward_time=0.01,
        beta_schedule=schedule,
    )
    delta_t = terminal_time
    delta_tau = beta_value * delta_t
    projector = s2_projector(terminal[0])
    tangent_increment = (
        delta_tau * (projector @ raw_score[0])
        + delta_tau**0.5 * (projector @ noise[0, 0])
    )
    expected = s2_exp(terminal[0], tangent_increment).reshape(1, 3)

    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-14)
    torch.testing.assert_close(
        received_times[0],
        torch.tensor([terminal_time], dtype=DTYPE),
        rtol=0,
        atol=0,
    )
