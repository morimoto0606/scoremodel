import copy
import json
import math

import pytest
import torch
import torch.nn as nn

from scripts import experiment_earthquake_teacher_compare_smoke as runner
from scoremodel_ext.malliavin.models import (
    NormalizedSkorokhodModel,
    learning_rate_for_update,
    train_mirafzali_skorokhod_net,
    update_ema_model,
)
from scoremodel_ext.manifold.s2_malliavin import s2_reverse_grw


DTYPE = torch.float64


def _assert_nested_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_equal(actual_item, expected_item)
    else:
        assert actual == expected


def _training_tensors(n_samples=10):
    generator = torch.Generator(device="cpu").manual_seed(2027)
    times = torch.linspace(0.05, 0.3, n_samples, dtype=DTYPE)
    points = torch.randn(n_samples, 3, generator=generator, dtype=DTYPE)
    points = points / torch.linalg.vector_norm(points, dim=1, keepdim=True)
    targets = torch.randn(n_samples, 3, generator=generator, dtype=DTYPE)
    return times, points, targets


def _train_small(**kwargs):
    times, points, targets = _training_tensors()
    return train_mirafzali_skorokhod_net(
        times,
        points,
        targets,
        n_epochs=2,
        batch_size=4,
        lr=2e-4,
        hidden=4,
        n_blocks=1,
        num_frequencies=2,
        device="cpu",
        return_history=True,
        return_training_state=True,
        **kwargs,
    )


def test_legacy_training_defaults_are_exactly_preserved():
    torch.manual_seed(41)
    implicit_model, implicit_history, implicit_state = _train_small()
    torch.manual_seed(41)
    explicit_model, explicit_history, explicit_state = _train_small(
        training_unit="epochs",
        updates=0,
        warmup_updates=0,
        lr_scheduler="constant",
        ema_rate=0.0,
        checkpoint_every_updates=0,
    )

    assert implicit_history == explicit_history
    assert implicit_state["legacy_training_path"] is True
    assert explicit_state["legacy_training_path"] is True
    assert implicit_state["actual_optimizer_updates"] == 2
    assert implicit_state["ema_model"] is None
    for key, value in implicit_model.state_dict().items():
        torch.testing.assert_close(
            value,
            explicit_model.state_dict()[key],
            rtol=0,
            atol=0,
        )


def test_update_training_has_exact_optimizer_step_count():
    torch.manual_seed(42)
    _, _, training_state = _train_small(
        training_unit="updates",
        updates=7,
    )
    assert training_state["actual_optimizer_updates"] == 7
    assert training_state["current_update"] == 7
    assert training_state["updates_per_epoch"] == 3
    assert training_state["effective_epochs"] == pytest.approx(7 / 3)


def test_periodic_checkpoint_callback_uses_completed_update_count():
    checkpoints = []
    torch.manual_seed(421)
    _train_small(
        training_unit="updates",
        updates=5,
        checkpoint_every_updates=2,
        checkpoint_callback=checkpoints.append,
    )
    assert [payload["current_update"] for payload in checkpoints] == [2, 4]
    assert all("optimizer_state_dict" in payload for payload in checkpoints)
    assert all("scheduler_state" in payload for payload in checkpoints)


def test_warmup_learning_rates_are_exact():
    base_lr = 2e-4
    actual = [
        learning_rate_for_update(
            index,
            total_updates=8,
            base_learning_rate=base_lr,
            warmup_updates=4,
            scheduler="constant",
        )
        for index in range(4)
    ]
    expected = [base_lr * factor for factor in (0.25, 0.5, 0.75, 1.0)]
    assert actual == expected


def test_cosine_learning_rate_start_middle_and_final():
    base_lr = 2e-4
    total_updates = 8
    warmup_updates = 2
    start = learning_rate_for_update(
        2,
        total_updates=total_updates,
        base_learning_rate=base_lr,
        warmup_updates=warmup_updates,
        scheduler="cosine",
    )
    middle = learning_rate_for_update(
        5,
        total_updates=total_updates,
        base_learning_rate=base_lr,
        warmup_updates=warmup_updates,
        scheduler="cosine",
    )
    final = learning_rate_for_update(
        7,
        total_updates=total_updates,
        base_learning_rate=base_lr,
        warmup_updates=warmup_updates,
        scheduler="cosine",
    )
    assert start == pytest.approx(base_lr)
    assert middle == pytest.approx(0.5 * base_lr)
    expected_final = 0.5 * base_lr * (1 + math.cos(5 / 6 * math.pi))
    assert final == pytest.approx(expected_final, abs=1e-12)


def test_ema_parameter_formula_and_buffer_copy():
    online = nn.Sequential(nn.Linear(1, 1, bias=False), nn.BatchNorm1d(1)).double()
    ema = copy.deepcopy(online)
    ema.requires_grad_(False)
    with torch.no_grad():
        online[0].weight.fill_(2.0)
        ema[0].weight.fill_(0.0)
        online[1].running_mean.fill_(3.0)
        online[1].running_var.fill_(4.0)

    update_ema_model(ema, online, 0.75)
    torch.testing.assert_close(
        ema[0].weight,
        torch.tensor([[0.5]], dtype=DTYPE),
        rtol=0,
        atol=1e-12,
    )
    torch.testing.assert_close(ema[1].running_mean, online[1].running_mean, rtol=0, atol=0)
    torch.testing.assert_close(ema[1].running_var, online[1].running_var, rtol=0, atol=0)

    with torch.no_grad():
        online[0].weight.fill_(4.0)
    update_ema_model(ema, online, 0.75)
    torch.testing.assert_close(
        ema[0].weight,
        torch.tensor([[1.375]], dtype=DTYPE),
        rtol=0,
        atol=1e-12,
    )


def test_ema_normalization_buffers_match_online_model():
    torch.manual_seed(43)
    online, _, training_state = _train_small(
        training_unit="updates",
        updates=2,
        ema_rate=0.9,
    )
    ema = training_state["ema_model"]
    assert ema is not None
    for key in ("x_mean", "x_std", "t_mean", "t_std", "y_mean", "y_std"):
        torch.testing.assert_close(
            getattr(ema, key),
            getattr(online, key),
            rtol=0,
            atol=0,
        )


def test_validation_and_reverse_model_selection():
    online = object()
    ema = object()
    assert runner.select_online_or_ema_model(
        online, ema, use_ema=False, purpose="validation"
    ) == (online, "online")
    assert runner.select_online_or_ema_model(
        online, ema, use_ema=True, purpose="reverse"
    ) == (ema, "ema")
    with pytest.raises(ValueError, match="EMA is disabled"):
        runner.select_online_or_ema_model(
            online, None, use_ema=True, purpose="validation"
        )


@pytest.mark.parametrize(
    ("use_ema", "expected_source", "expected_loss"),
    [(False, "online", 1.25), (True, "ema", 2.5)],
)
def test_validation_evaluates_only_selected_model(
    monkeypatch,
    use_ema,
    expected_source,
    expected_loss,
):
    online = object()
    ema = object()
    calls = []

    def fake_evaluate(model, dataset, *, teacher):
        calls.append((model, dataset, teacher))
        return 1.25 if model is online else 2.5

    monkeypatch.setattr(runner, "evaluate_dataset_loss", fake_evaluate)
    dataset = {"sentinel": torch.tensor(1.0)}
    loss, online_loss, ema_loss, source = (
        runner.evaluate_selected_validation_loss(
            online,
            ema,
            dataset,
            teacher="heat",
            use_ema=use_ema,
        )
    )

    expected_model = ema if use_ema else online
    assert calls == [(expected_model, dataset, "heat")]
    assert loss == expected_loss
    assert source == expected_source
    assert online_loss == (None if use_ema else expected_loss)
    assert ema_loss == (expected_loss if use_ema else None)


def test_checkpoint_round_trip_and_selected_ema_state(tmp_path):
    torch.manual_seed(44)
    online, _, training_state = _train_small(
        training_unit="updates",
        updates=2,
        warmup_updates=1,
        lr_scheduler="cosine",
        ema_rate=0.9,
    )
    ema = training_state["ema_model"]
    payload = runner.build_model_checkpoint_payload(
        selected_model=ema,
        online_model=online,
        ema_model=ema,
        model_source="ema",
        teacher="heat",
        training_path="direct_score",
        hidden=4,
        n_blocks=1,
        num_frequencies=2,
        dtype="float64",
        training_state=training_state,
        training_unit="updates",
        requested_epochs=2,
        requested_updates=2,
        base_learning_rate=1e-3,
        warmup_updates=1,
        lr_scheduler="cosine",
        ema_rate=0.9,
        use_ema_for_validation=True,
        use_ema_for_reverse=True,
        checkpoint_every_updates=1,
        beta_schedule="legacy-unit",
        beta_0=0.001,
        beta_f=5.0,
        beta_t0=0.0,
        beta_tf=1.0,
    )
    checkpoint_path = tmp_path / "model.pt"
    torch.save(payload, checkpoint_path)
    restored = torch.load(checkpoint_path, map_location="cpu")

    assert restored["model_source"] == "ema"
    assert restored["current_update"] == 2
    _assert_nested_equal(restored["scheduler_state"], training_state["scheduler_state"])
    _assert_nested_equal(
        restored["optimizer_state_dict"],
        training_state["optimizer_state_dict"],
    )
    for key, value in restored["state_dict"].items():
        torch.testing.assert_close(value, restored["ema_state_dict"][key], rtol=0, atol=0)
    for key, value in online.state_dict().items():
        torch.testing.assert_close(
            value,
            restored["online_state_dict"][key],
            rtol=0,
            atol=0,
        )


def test_ema_saved_model_replays_fixed_output_and_reverse_first_step(tmp_path):
    torch.manual_seed(45)
    online, _, training_state = _train_small(
        training_unit="updates",
        updates=2,
        ema_rate=0.9,
    )
    ema = training_state["ema_model"]
    payload = runner.build_model_checkpoint_payload(
        selected_model=ema,
        online_model=online,
        ema_model=ema,
        model_source="ema",
        teacher="heat",
        training_path="direct_score",
        hidden=4,
        n_blocks=1,
        num_frequencies=2,
        dtype="float64",
        training_state=training_state,
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
        beta_schedule="legacy-unit",
        beta_0=0.001,
        beta_f=5.0,
        beta_t0=0.0,
        beta_tf=1.0,
    )
    model_path = tmp_path / "model.pt"
    torch.save(payload, model_path)
    run_config = {
        "teacher": "heat",
        "hidden": 4,
        "n_blocks": 1,
        "num_frequencies": 2,
        "dtype": "float64",
        "use_ema_for_reverse": True,
    }
    (tmp_path / "run_config.json").write_text(json.dumps(run_config))
    replay = runner.build_model_from_run_config(model_path, run_config, device="cpu")

    times = torch.tensor([0.2], dtype=DTYPE)
    points = torch.tensor([[0.0, 0.0, 1.0]], dtype=DTYPE)
    with torch.no_grad():
        expected_output = ema(times, points)
        replay_output = replay(times, points)
    torch.testing.assert_close(replay_output, expected_output, rtol=0, atol=1e-12)

    reverse_noise = torch.tensor([[[0.2, -0.1, 0.3]]], dtype=DTYPE)
    expected_final, expected_first = s2_reverse_grw(
        points,
        runner.build_score_fn(ema),
        terminal_time=0.2,
        n_steps=1,
        standard_noise=reverse_noise,
        return_first_step=True,
    )
    replay_final, replay_first = s2_reverse_grw(
        points,
        runner.build_score_fn(replay),
        terminal_time=0.2,
        n_steps=1,
        standard_noise=reverse_noise,
        return_first_step=True,
    )
    torch.testing.assert_close(replay_first, expected_first, rtol=0, atol=1e-12)
    torch.testing.assert_close(replay_final, expected_final, rtol=0, atol=1e-12)
