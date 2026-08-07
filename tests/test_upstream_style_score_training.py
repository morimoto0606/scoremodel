import pytest
import torch
from torch import nn

from scoremodel_ext.manifold.beta_schedule import LinearBetaSchedule
from scoremodel_ext.manifold.upstream_style_score import (
    UpstreamStyleScoreModel,
    build_upstream_style_score_model,
    train_s2_upstream_style_score_model,
    upstream_score_standard_deviation,
    upstream_style_score_loss,
)
from scripts import experiment_earthquake_teacher_compare_smoke as runner
from scripts.earthquake_heat_upstream_style_training import configured_arguments
from scripts.earthquake_malliavin_upstream_style_training import (
    configured_arguments as configured_malliavin_arguments,
)


DTYPE = torch.float64


class _FixedRawNetwork(nn.Module):
    def __init__(self, raw_output):
        super().__init__()
        self.register_buffer("value", raw_output)

    def forward(self, time, points):
        del time, points
        return self.value


def _model(raw_output):
    return UpstreamStyleScoreModel(
        _FixedRawNetwork(raw_output),
        torch.zeros(1, 3, dtype=DTYPE),
        torch.ones(1, 3, dtype=DTYPE),
        torch.zeros(1, 1, dtype=DTYPE),
        torch.ones(1, 1, dtype=DTYPE),
        beta_0=0.001,
        beta_f=5.0,
        beta_t0=0.0,
        beta_tf=1.0,
    )


def test_raw_output_is_divided_by_sigma_once_for_effective_score():
    times = torch.tensor([0.01, 0.5], dtype=DTYPE)
    points = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=DTYPE
    )
    schedule = LinearBetaSchedule(beta_0=0.001, beta_f=5.0, t0=0.0, tf=1.0)
    sigma = upstream_score_standard_deviation(times, schedule)
    effective_score = torch.tensor(
        [[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]], dtype=DTYPE
    )
    raw_output = sigma[:, None] * effective_score
    model = _model(raw_output)

    torch.testing.assert_close(model.raw_output(times, points), raw_output)
    torch.testing.assert_close(model(times, points), effective_score)


def test_loss_is_sigma_weighted_effective_score_error():
    times = torch.tensor([0.01, 0.5], dtype=DTYPE)
    schedule = LinearBetaSchedule(beta_0=0.001, beta_f=5.0, t0=0.0, tf=1.0)
    sigma = upstream_score_standard_deviation(times, schedule)
    predicted_score = torch.tensor(
        [[1.0, 2.0, 0.0], [0.5, -1.0, 2.0]], dtype=DTYPE
    )
    teacher_score = torch.tensor(
        [[0.0, 2.0, 1.0], [1.5, -1.0, 0.0]], dtype=DTYPE
    )
    raw_output = sigma[:, None] * predicted_score

    loss = upstream_style_score_loss(
        raw_output, teacher_score, times, schedule
    )
    expected = (
        sigma[:, None] * (predicted_score - teacher_score)
    ).square().sum(dim=1).mean()

    torch.testing.assert_close(loss, expected)
    unweighted = (predicted_score - teacher_score).square().sum(dim=1).mean()
    assert float(loss) != pytest.approx(float(unweighted))


def test_sigma_matches_upstream_log_mean_coeff_definition():
    times = torch.tensor(
        [0.001, 0.01, 0.05, 0.1, 0.5, 0.9, 1.0], dtype=DTYPE
    )
    beta_0 = 0.001
    beta_f = 5.0
    t0 = 0.0
    tf = 1.0
    normalized_time = (times - t0) / (tf - t0)
    # Exact reference transcription of Upstream LinearBetaSchedule and
    # Brownian.marginal_prob:
    #   log_mean_coeff = -0.5 * (0.5*u^2*(beta_f-beta_0) + u*beta_0)
    #   std = sqrt(1 - exp(2*log_mean_coeff))
    upstream_log_mean_coeff = -0.5 * (
        0.5 * normalized_time.square() * (beta_f - beta_0)
        + normalized_time * beta_0
    )
    upstream_std = torch.sqrt(1.0 - torch.exp(2.0 * upstream_log_mean_coeff))

    schedule = LinearBetaSchedule(
        beta_0=beta_0, beta_f=beta_f, t0=t0, tf=tf
    )
    helper_sigma = upstream_score_standard_deviation(times, schedule)
    model_sigma = _model(torch.zeros(7, 3, dtype=DTYPE)).score_standard_deviation(
        times
    )

    torch.testing.assert_close(helper_sigma, upstream_std, rtol=1e-15, atol=0.0)
    torch.testing.assert_close(model_sigma, upstream_std, rtol=1e-15, atol=0.0)
    torch.testing.assert_close(model_sigma, helper_sigma, rtol=0.0, atol=0.0)


def test_experiment_entry_point_fixes_heat_upstream_style_path():
    arguments = configured_arguments(["--updates", "2", "--output-dir", "out"])

    assert arguments[:4] == [
        "--teacher",
        "heat",
        "--score-parameterization",
        "upstream_scaled_score",
    ]
    assert arguments.count("--updates") == 1
    assert arguments[arguments.index("--updates") + 1] == "2"


def test_experiment_entry_point_rejects_parameterization_override():
    with pytest.raises(ValueError, match="fixes --teacher heat"):
        configured_arguments(["--teacher", "varadhan"])
    with pytest.raises(ValueError, match="fixes --score-parameterization"):
        configured_arguments(["--score-parameterization", "effective_score"])


def test_malliavin_entry_point_fixes_upstream_scaled_score_conditions():
    arguments = configured_malliavin_arguments(
        ["--updates", "2", "--output-dir", "out"]
    )

    assert arguments[:4] == [
        "--teacher",
        "malliavin",
        "--score-parameterization",
        "upstream_scaled_score",
    ]
    expected_values = {
        "--batch-size": "512",
        "--hidden": "1024",
        "--n-blocks": "6",
        "--num-frequencies": "16",
        "--learning-rate": "2e-4",
        "--dtype": "float64",
        "--beta-schedule": "linear",
        "--beta-0": "0.001",
        "--beta-f": "5.0",
        "--minimum-time": "0.001",
        "--maximum-time": "1.0",
        "--ema-rate": "0.999",
    }
    for option, expected in expected_values.items():
        assert arguments[arguments.index(option) + 1] == expected
    assert "--use-ema-for-validation" in arguments
    assert "--use-ema-for-reverse" in arguments


def test_malliavin_scaled_validation_uses_score_target_not_skorokhod():
    times = torch.tensor([0.01, 0.5], dtype=DTYPE)
    points = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=DTYPE
    )
    schedule = LinearBetaSchedule(beta_0=0.001, beta_f=5.0, t0=0.0, tf=1.0)
    teacher_score = torch.tensor(
        [[0.2, -0.1, 0.0], [0.4, 0.3, -0.2]], dtype=DTYPE
    )
    effective_prediction = teacher_score + 0.25
    sigma = upstream_score_standard_deviation(times, schedule)
    model = _model(sigma[:, None] * effective_prediction)
    dataset = {
        "time": times,
        "endpoint": points,
        "score_target": teacher_score,
        "skorokhod": torch.full_like(teacher_score, 1e6),
    }

    actual = runner.evaluate_dataset_loss(model, dataset, teacher="malliavin")
    expected = upstream_style_score_loss(
        sigma[:, None] * effective_prediction,
        teacher_score,
        times,
        schedule,
    )
    assert actual == pytest.approx(float(expected), rel=0.0, abs=1e-15)


def test_malliavin_scaled_checkpoint_reload_preserves_effective_output(tmp_path):
    torch.manual_seed(31)
    schedule = LinearBetaSchedule(beta_0=0.001, beta_f=5.0, t0=0.0, tf=1.0)
    model = build_upstream_style_score_model(
        x_mean=torch.tensor([[0.1, -0.2, 0.3]], dtype=DTYPE),
        x_std=torch.tensor([[0.7, 0.8, 0.9]], dtype=DTYPE),
        t_mean=torch.tensor([[0.4]], dtype=DTYPE),
        t_std=torch.tensor([[0.2]], dtype=DTYPE),
        hidden=8,
        n_blocks=1,
        num_frequencies=2,
        beta_schedule=schedule,
        device="cpu",
        dtype=DTYPE,
    )
    model.eval()
    checkpoint = {
        "teacher": "malliavin",
        "training_path": "upstream_scaled_score",
        "score_parameterization": "upstream_scaled_score",
        "state_dict": model.state_dict(),
        "hidden": 8,
        "n_blocks": 1,
        "num_frequencies": 2,
        "dtype": "float64",
        "beta_0": 0.001,
        "beta_f": 5.0,
        "beta_t0": 0.0,
        "beta_tf": 1.0,
    }
    path = tmp_path / "model.pt"
    torch.save(checkpoint, path)
    restored = runner.build_model_from_training_checkpoint(path, device="cpu")
    times = torch.tensor([0.01, 0.25, 0.9], dtype=DTYPE)
    points = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=DTYPE,
    )
    with torch.no_grad():
        expected = model(times, points)
        actual = restored(times, points)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_new_training_path_checkpoint_round_trip(tmp_path):
    torch.manual_seed(7)
    schedule = LinearBetaSchedule(beta_0=0.001, beta_f=5.0, t0=0.0, tf=1.0)
    times = torch.tensor([0.01, 0.1, 0.5, 0.9], dtype=DTYPE)
    endpoints = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [2.0**-0.5, 2.0**-0.5, 0.0],
        ],
        dtype=DTYPE,
    )
    targets = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, -0.5, 0.0],
            [-1.0, 1.0, 0.0],
        ],
        dtype=DTYPE,
    )
    model, _, state = train_s2_upstream_style_score_model(
        {"time": times, "endpoint": endpoints, "score_target": targets},
        beta_schedule=schedule,
        n_epochs=2,
        batch_size=2,
        learning_rate=1e-3,
        hidden=8,
        n_blocks=1,
        num_frequencies=2,
        device="cpu",
        return_history=True,
        training_unit="updates",
        updates=2,
        ema_rate=0.9,
        return_training_state=True,
    )
    ema_model = state["ema_model"]
    payload = runner.build_model_checkpoint_payload(
        selected_model=ema_model,
        online_model=model,
        ema_model=ema_model,
        model_source="ema",
        teacher="heat",
        training_path="upstream_scaled_score",
        hidden=8,
        n_blocks=1,
        num_frequencies=2,
        dtype="float64",
        training_state=state,
        training_unit="updates",
        requested_epochs=2,
        requested_updates=2,
        base_learning_rate=1e-3,
        warmup_updates=0,
        lr_scheduler="constant",
        ema_rate=0.9,
        use_ema_for_validation=True,
        use_ema_for_reverse=True,
        checkpoint_every_updates=0,
        beta_schedule="linear",
        beta_0=0.001,
        beta_f=5.0,
        beta_t0=0.0,
        beta_tf=1.0,
    )
    path = tmp_path / "model.pt"
    torch.save(payload, path)

    restored = runner.build_model_from_training_checkpoint(path, device="cpu")
    traced, trace, traced_inner, prefix = (
        runner.build_model_from_training_checkpoint_with_normalization_trace(
            path, device="cpu"
        )
    )
    with torch.no_grad():
        expected = ema_model(times, endpoints)
        actual = restored(times, endpoints)
        traced_actual = traced(times, endpoints)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(traced_actual, expected, rtol=0.0, atol=0.0)
    assert traced_inner is traced
    assert prefix == ""
    assert trace["stages"][0]["all_normalization_buffers_exact"] is True
    assert payload["score_parameterization"] == "upstream_scaled_score"
