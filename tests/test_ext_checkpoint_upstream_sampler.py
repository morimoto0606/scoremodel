import torch

from scripts.run_ext_checkpoint_with_upstream_sampler import (
    make_reverse_inputs,
    run_upstream_compatible_reverse,
)
from scoremodel_ext.manifold.beta_schedule import LinearBetaSchedule


DTYPE = torch.float64


def _schedule() -> LinearBetaSchedule:
    return LinearBetaSchedule(beta_0=0.001, beta_f=5.0, t0=0.0, tf=1.0)


def test_upstream_compatible_reverse_preserves_s2_norm():
    terminal, noise = make_reverse_inputs(
        8,
        dtype=DTYPE,
        device="cpu",
        seed=17,
        n_steps=6,
    )

    result = run_upstream_compatible_reverse(
        terminal,
        lambda time, points: torch.zeros_like(points),
        noise,
        beta_schedule=_schedule(),
        terminal_time=1.0,
        epsilon=0.001,
        n_steps=6,
    )

    expected = torch.ones(8, dtype=DTYPE)
    torch.testing.assert_close(
        torch.linalg.vector_norm(result.final_samples, dim=1),
        expected,
        rtol=0,
        atol=1e-12,
    )


def test_upstream_compatible_reverse_is_deterministic_for_seed():
    first_inputs = make_reverse_inputs(
        5, dtype=DTYPE, device="cpu", seed=2026, n_steps=4
    )
    second_inputs = make_reverse_inputs(
        5, dtype=DTYPE, device="cpu", seed=2026, n_steps=4
    )
    for first, second in zip(first_inputs, second_inputs):
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def score_fn(time, points):
        return torch.stack((0.2 * time, -0.1 * time, 0.05 * time), dim=1)

    first = run_upstream_compatible_reverse(
        first_inputs[0],
        score_fn,
        first_inputs[1],
        beta_schedule=_schedule(),
        n_steps=4,
    )
    second = run_upstream_compatible_reverse(
        second_inputs[0],
        score_fn,
        second_inputs[1],
        beta_schedule=_schedule(),
        n_steps=4,
    )
    torch.testing.assert_close(first.trajectory, second.trajectory, rtol=0, atol=0)


def test_effective_score_is_not_divided_by_upstream_standard_deviation():
    terminal = torch.tensor([[0.0, 0.0, 1.0]], dtype=DTYPE)
    noise = torch.zeros(1, 1, 3, dtype=DTYPE)
    checkpoint_output = torch.tensor([[0.3, -0.4, 0.2]], dtype=DTYPE)

    result = run_upstream_compatible_reverse(
        terminal,
        lambda time, points: checkpoint_output.expand_as(points),
        noise,
        beta_schedule=_schedule(),
        terminal_time=1.0,
        epsilon=0.5,
        n_steps=1,
    )

    expected_norm = torch.linalg.vector_norm(checkpoint_output, dim=1).reshape(1, 1)
    torch.testing.assert_close(result.network_output_norm, expected_norm, rtol=0, atol=0)
    torch.testing.assert_close(result.score_norm, expected_norm, rtol=0, atol=0)
    incorrectly_rescaled_norm = expected_norm / result.score_std
    assert not torch.allclose(result.score_norm, incorrectly_rescaled_norm)
