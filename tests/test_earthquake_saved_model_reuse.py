import json
import sys

import pytest
import torch

from scoremodel_ext.malliavin.models import (
    MirafzaliSkorokhodNet,
    NormalizedSkorokhodModel,
)
from scoremodel_ext.manifold.s2_malliavin import S2SkorokhodScoreModel
from scripts import experiment_earthquake_teacher_compare_smoke as runner


def _saved_model(tmp_path, teacher):
    config = {
        "teacher": teacher,
        "hidden": 7,
        "n_blocks": 2,
        "num_frequencies": 5,
        "dtype": "float64",
    }
    network = MirafzaliSkorokhodNet(
        x_dim=3,
        out_dim=3,
        hidden=config["hidden"],
        n_blocks=config["n_blocks"],
        num_frequencies=config["num_frequencies"],
    ).double()
    vector_zero = torch.zeros(1, 3, dtype=torch.float64)
    vector_one = torch.ones(1, 3, dtype=torch.float64)
    time_zero = torch.zeros(1, 1, dtype=torch.float64)
    time_one = torch.ones(1, 1, dtype=torch.float64)
    model = NormalizedSkorokhodModel(
        network,
        vector_zero,
        vector_one,
        time_zero,
        time_one,
        vector_zero,
        vector_one,
    ).double()
    if teacher == "malliavin":
        model = S2SkorokhodScoreModel(model)
    model_path = tmp_path / "model.pt"
    torch.save({"state_dict": model.state_dict()}, model_path)
    (tmp_path / "run_config.json").write_text(json.dumps(config))
    return model_path, config, model


@pytest.mark.parametrize("teacher", ["heat", "malliavin"])
def test_saved_model_structure_is_rebuilt_from_run_config(tmp_path, teacher):
    model_path, config, expected = _saved_model(tmp_path, teacher)

    loaded = runner.build_model_from_run_config(
        model_path,
        config,
        device="cpu",
    )

    assert type(loaded) is type(expected)
    assert loaded.state_dict().keys() == expected.state_dict().keys()
    for name, value in expected.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], value)


def test_shared_reverse_noise_is_saved_at_1000_steps_and_coarsened(tmp_path):
    pool_path = tmp_path / "shared.pt"
    first = runner.maybe_load_or_create_shared_reverse_noise(
        path=pool_path,
        output_path=tmp_path / "copy_10.pt",
        reverse_steps=10,
        n_generated_samples=3,
        dtype=torch.float64,
        device="cpu",
        seed=11,
    )
    second = runner.maybe_load_or_create_shared_reverse_noise(
        path=pool_path,
        output_path=tmp_path / "copy_20.pt",
        reverse_steps=20,
        n_generated_samples=3,
        dtype=torch.float64,
        device="cpu",
        seed=999,
    )
    pool = torch.load(pool_path, map_location="cpu")

    assert pool.shape == (1000, 3, 3)
    assert torch.load(tmp_path / "copy_10.pt").shape == (10, 3, 3)
    assert torch.load(tmp_path / "copy_20.pt").shape == (20, 3, 3)
    assert first.shape == (10, 3, 3)
    assert second.shape == (20, 3, 3)
    expected_first_increment = (second[0] + second[1]) / (2.0**0.5)
    assert torch.allclose(first[0], expected_first_increment)


def _interpolated_fine_brownian_path(pool, reverse_steps, terminal_time):
    fine_steps = pool.shape[0]
    fine_path = torch.cat(
        (
            torch.zeros(1, *pool.shape[1:], dtype=pool.dtype),
            torch.cumsum(pool, dim=0),
        ),
        dim=0,
    ) * (terminal_time / fine_steps) ** 0.5
    positions = (
        torch.arange(reverse_steps + 1, dtype=torch.float64)
        * fine_steps
        / reverse_steps
    )
    lower = torch.floor(positions).long()
    upper = torch.ceil(positions).long().clamp_max(fine_steps)
    fraction = (positions - lower).to(pool.dtype)
    return (
        fine_path[lower] * (1.0 - fraction[:, None, None])
        + fine_path[upper] * fraction[:, None, None]
    )


@pytest.mark.parametrize("reverse_steps", [128, 256, 512, 1000])
def test_coarse_noise_matches_interpolated_fine_brownian_increments(
    reverse_steps,
):
    terminal_time = 0.3
    generator = torch.Generator(device="cpu").manual_seed(19)
    pool = torch.randn(
        runner.MAX_REVERSE_NOISE_STEPS,
        4,
        3,
        generator=generator,
        dtype=torch.float64,
    )

    coarse = runner.aggregate_reverse_noise_pool(
        pool,
        reverse_steps=reverse_steps,
    )
    interpolated_path = _interpolated_fine_brownian_path(
        pool,
        reverse_steps,
        terminal_time,
    )
    coarse_brownian_increments = (
        (terminal_time / reverse_steps) ** 0.5 * coarse
    )

    assert coarse.shape == (reverse_steps, 4, 3)
    assert torch.isfinite(coarse).all()
    torch.testing.assert_close(
        coarse_brownian_increments,
        interpolated_path[1:] - interpolated_path[:-1],
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        coarse_brownian_increments.sum(dim=0),
        (terminal_time / runner.MAX_REVERSE_NOISE_STEPS) ** 0.5
        * pool.sum(dim=0),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("reverse_steps", [10, 125, 250, 500, 1000])
def test_divisor_step_counts_are_exact_normalized_fine_increment_sums(
    reverse_steps,
):
    generator = torch.Generator(device="cpu").manual_seed(23)
    pool = torch.randn(1000, 2, 3, generator=generator, dtype=torch.float64)
    block_size = 1000 // reverse_steps
    expected = (
        pool.reshape(reverse_steps, block_size, 2, 3).sum(dim=1)
        * (reverse_steps / 1000) ** 0.5
    )

    actual = runner.aggregate_reverse_noise_pool(
        pool,
        reverse_steps=reverse_steps,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("reverse_steps", [128, 256, 512, 1000])
def test_shared_noise_is_reproducible_for_ablation_step_counts(
    tmp_path,
    reverse_steps,
):
    first = runner.maybe_load_or_create_shared_reverse_noise(
        path=tmp_path / f"pool_a_{reverse_steps}.pt",
        output_path=tmp_path / f"coarse_a_{reverse_steps}.pt",
        reverse_steps=reverse_steps,
        n_generated_samples=2,
        dtype=torch.float64,
        device="cpu",
        seed=31,
    )
    second = runner.maybe_load_or_create_shared_reverse_noise(
        path=tmp_path / f"pool_b_{reverse_steps}.pt",
        output_path=tmp_path / f"coarse_b_{reverse_steps}.pt",
        reverse_steps=reverse_steps,
        n_generated_samples=2,
        dtype=torch.float64,
        device="cpu",
        seed=31,
    )

    assert torch.equal(first, second)


def test_skip_training_requires_model_path(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "experiment_earthquake_teacher_compare_smoke.py",
            "--output-dir",
            str(tmp_path),
            "--skip-training",
        ],
    )
    with pytest.raises(SystemExit):
        runner.parse_args()
