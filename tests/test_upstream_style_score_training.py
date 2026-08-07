import pytest
import torch
from torch import nn

from scoremodel_ext.manifold.beta_schedule import LinearBetaSchedule
from scoremodel_ext.manifold.upstream_style_score import (
    UpstreamStyleScoreModel,
    train_s2_upstream_style_score_model,
    upstream_score_standard_deviation,
    upstream_style_score_loss,
)
from scripts import experiment_earthquake_teacher_compare_smoke as runner
from scripts.earthquake_heat_upstream_style_training import configured_arguments


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
