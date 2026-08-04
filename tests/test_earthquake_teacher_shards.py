import copy
from types import SimpleNamespace

import pytest
import torch

from scripts.combine_earthquake_teacher_shards import combine_teacher_shards
from scripts import experiment_earthquake_teacher_compare_smoke as runner
from scoremodel_ext.manifold.s2_malliavin import (
    s2_batched_discrete_malliavin_teacher,
    s2_discrete_malliavin_teacher,
)


DTYPE = torch.float64
REQUIRED_DETAIL_KEYS = (
    "endpoint",
    "endpoint_jacobian",
    "covariance",
    "covering",
    "divergence_term",
    "skorokhod",
    "score_weight",
)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_torch_func_batched_teacher_matches_scalar_float64(batch_size):
    generator = torch.Generator(device="cpu").manual_seed(314)
    initial_points = torch.randn(2, 3, generator=generator, dtype=DTYPE)
    initial_points = initial_points / torch.linalg.vector_norm(
        initial_points, dim=1, keepdim=True
    )
    noises = torch.randn(2, 2, 3, generator=generator, dtype=DTYPE)
    times = torch.tensor([0.1, 0.25], dtype=DTYPE)

    dataset, details, effective = runner.build_malliavin_teacher_dataset_batched(
        initial_points=initial_points,
        times=times,
        noises=noises,
        batch_size=batch_size,
        covariance_regularization=1e-6,
    )
    scalar = [
        s2_discrete_malliavin_teacher(
            initial_points[index],
            noises[index],
            float(times[index]),
            covariance_regularization=1e-6,
        )
        for index in range(2)
    ]
    assert sum(effective) == 2
    torch.testing.assert_close(dataset["initial_point"], initial_points, rtol=0, atol=0)
    torch.testing.assert_close(dataset["time"], times, rtol=0, atol=0)
    torch.testing.assert_close(dataset["noise"], noises, rtol=0, atol=0)
    for key in REQUIRED_DETAIL_KEYS:
        reference = torch.stack([getattr(item, key) for item in scalar])
        candidate = details[key]
        max_abs_error = float(torch.max(torch.abs(candidate - reference)))
        assert max_abs_error <= 1e-12, (batch_size, key, max_abs_error)
        torch.testing.assert_close(candidate, reference, rtol=0.0, atol=1e-12)


def test_batched_adapter_batch_size_one_matches_scalar():
    initial = torch.tensor([[0.0, 0.0, 1.0]], dtype=DTYPE)
    noise = torch.tensor([[[0.2, -0.1, 0.3], [-0.4, 0.5, 0.1]]], dtype=DTYPE)
    time = torch.tensor([0.2], dtype=DTYPE)
    batched = s2_batched_discrete_malliavin_teacher(initial, noise, time)
    scalar = s2_discrete_malliavin_teacher(initial[0], noise[0], 0.2)
    for key in REQUIRED_DETAIL_KEYS:
        torch.testing.assert_close(
            getattr(batched, key)[0],
            getattr(scalar, key),
            rtol=0.0,
            atol=1e-12,
        )


def test_oom_fallback_retries_same_samples_at_half_batch(monkeypatch):
    initial = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=DTYPE
    )
    noise = torch.tensor(
        [
            [[0.2, -0.1, 0.3], [-0.4, 0.5, 0.1]],
            [[0.1, 0.2, -0.2], [0.3, -0.1, 0.4]],
        ],
        dtype=DTYPE,
    )
    time = torch.tensor([0.1, 0.2], dtype=DTYPE)
    original = runner.s2_batched_discrete_malliavin_teacher
    attempted_sizes = []

    def fail_once_for_two(initial_points, noises, times, **kwargs):
        attempted_sizes.append(initial_points.shape[0])
        if initial_points.shape[0] == 2:
            raise RuntimeError("CUDA out of memory (synthetic test)")
        return original(initial_points, noises, times, **kwargs)

    monkeypatch.setattr(
        runner,
        "s2_batched_discrete_malliavin_teacher",
        fail_once_for_two,
    )
    dataset, _, effective = runner.build_malliavin_teacher_dataset_batched(
        initial_points=initial,
        times=time,
        noises=noise,
        batch_size=2,
        covariance_regularization=1e-6,
    )

    assert attempted_sizes == [2, 1, 1]
    assert effective == [1, 1]
    torch.testing.assert_close(dataset["initial_point"], initial, rtol=0, atol=0)
    torch.testing.assert_close(dataset["noise"], noise, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_torch_func_batched_teacher_matches_scalar_cuda_float64():
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(2718)
    initial_points = torch.randn(
        4, 3, generator=generator, dtype=DTYPE, device=device
    )
    initial_points = initial_points / torch.linalg.vector_norm(
        initial_points, dim=1, keepdim=True
    )
    noises = torch.randn(
        4, 2, 3, generator=generator, dtype=DTYPE, device=device
    )
    times = torch.tensor([0.05, 0.1, 0.2, 0.3], dtype=DTYPE, device=device)
    batched = s2_batched_discrete_malliavin_teacher(
        initial_points,
        noises,
        times,
    )
    for index in range(4):
        scalar = s2_discrete_malliavin_teacher(
            initial_points[index],
            noises[index],
            float(times[index].detach().cpu()),
        )
        for key in REQUIRED_DETAIL_KEYS:
            candidate = getattr(batched, key)[index]
            reference = getattr(scalar, key)
            max_abs_error = float(
                torch.max(torch.abs(candidate - reference)).detach().cpu()
            )
            assert max_abs_error <= 1e-12, (index, key, max_abs_error)
            torch.testing.assert_close(
                candidate,
                reference,
                rtol=0.0,
                atol=1e-12,
            )


def test_teacher_shard_worker_loads_saved_inputs_without_rng(monkeypatch, tmp_path):
    train_size = 2
    validation_size = 1
    n_steps = 2
    initial_payload = {
        "train_initial_points": torch.tensor(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=DTYPE
        ),
        "validation_initial_points": torch.tensor(
            [[0.0, 1.0, 0.0]], dtype=DTYPE
        ),
    }
    train_times = torch.tensor([0.1, 0.2], dtype=DTYPE)
    validation_times = torch.tensor([0.3], dtype=DTYPE)
    noises = {
        "train_noises": torch.arange(
            train_size * n_steps * 3, dtype=DTYPE
        ).reshape(train_size, n_steps, 3)
        / 10.0,
        "validation_noises": torch.arange(
            validation_size * n_steps * 3, dtype=DTYPE
        ).reshape(validation_size, n_steps, 3)
        / 20.0,
    }
    initial_path = tmp_path / "teacher_initial_points.pt"
    train_times_path = tmp_path / "time_samples.pt"
    validation_times_path = tmp_path / "validation_time_samples.pt"
    noises_path = tmp_path / "teacher_noises.pt"
    torch.save(initial_payload, initial_path)
    torch.save(train_times, train_times_path)
    torch.save(validation_times, validation_times_path)
    torch.save(noises, noises_path)

    def fail_rng(*args, **kwargs):
        raise AssertionError("shard worker must not call an RNG API")

    for name in ("rand", "randn", "randint", "randperm", "multinomial"):
        monkeypatch.setattr(torch, name, fail_rng)
    args = SimpleNamespace(
        dtype="float64",
        device="cpu",
        teacher_initial_points_path=initial_path,
        time_samples_path=train_times_path,
        validation_time_samples_path=validation_times_path,
        teacher_noises_path=noises_path,
        train_size=train_size,
        validation_size=validation_size,
        n_steps=n_steps,
        teacher_start_index=0,
        teacher_end_index=3,
        covariance_regularization=1e-6,
    )
    output_dir = tmp_path / "worker"
    output_dir.mkdir()
    runner.run_teacher_dataset_shard(args, output_dir=output_dir, log=lambda _: None)

    payload = torch.load(
        output_dir / "teacher_dataset_shard_000000_000003.pt"
    )
    assert payload["start"] == 0
    assert payload["end"] == 3
    assert payload["dataset_keys"] == [
        "initial_point",
        "time",
        "noise",
        "endpoint",
        "score_target",
        "skorokhod",
    ]
    assert payload["dtype"] == "float64"
    torch.testing.assert_close(
        payload["dataset"]["noise"][:train_size],
        noises["train_noises"],
        rtol=0.0,
        atol=0.0,
    )


def test_float64_scalar_full_batch_matches_four_combined_shards(tmp_path):
    train_size = 32
    validation_size = 8
    total_size = train_size + validation_size
    n_steps = 4
    generator = torch.Generator(device="cpu").manual_seed(1729)
    initial_points = torch.randn(
        total_size, 3, generator=generator, dtype=DTYPE
    )
    initial_points = initial_points / torch.linalg.vector_norm(
        initial_points, dim=1, keepdim=True
    )
    times = torch.linspace(0.05, 0.3, total_size, dtype=DTYPE)
    noises = torch.randn(
        total_size,
        n_steps,
        3,
        generator=generator,
        dtype=DTYPE,
    )

    scalar_full = runner.build_malliavin_teacher_shard(
        initial_points=initial_points,
        times=times,
        noises=noises,
        start=0,
        end=total_size,
        train_size=train_size,
        validation_size=validation_size,
        covariance_regularization=1e-6,
    )
    shard_paths = []
    for shard_index, (start, end) in enumerate(
        ((0, 10), (10, 20), (20, 30), (30, 40))
    ):
        payload = runner.build_malliavin_teacher_shard(
            initial_points=initial_points,
            times=times,
            noises=noises,
            start=start,
            end=end,
            train_size=train_size,
            validation_size=validation_size,
            covariance_regularization=1e-6,
        )
        path = tmp_path / f"shard_{shard_index}.pt"
        torch.save(payload, path)
        shard_paths.append(path)

    result = combine_teacher_shards(
        shard_paths,
        output_dir=tmp_path / "combined",
    )
    combined_dataset = {
        key: torch.cat(
            (result["train_dataset"][key], result["validation_dataset"][key])
        )
        for key in scalar_full["dataset_keys"]
    }
    combined_details = {
        key: torch.cat(
            (result["train_details"][key], result["validation_details"][key])
        )
        for key in scalar_full["detail_keys"]
    }

    assert tuple(combined_details) == REQUIRED_DETAIL_KEYS
    for key in scalar_full["dataset_keys"]:
        reference = scalar_full["dataset"][key]
        candidate = combined_dataset[key]
        assert candidate.shape == reference.shape
        assert candidate.dtype == reference.dtype == DTYPE
        max_abs_error = float(torch.max(torch.abs(candidate - reference)))
        assert max_abs_error <= 1e-12, (key, max_abs_error)
        torch.testing.assert_close(candidate, reference, rtol=0.0, atol=1e-12)
    for key in REQUIRED_DETAIL_KEYS:
        reference = scalar_full["teacher_details"][key]
        candidate = combined_details[key]
        assert candidate.shape == reference.shape
        assert candidate.dtype == reference.dtype == DTYPE
        max_abs_error = float(torch.max(torch.abs(candidate - reference)))
        assert max_abs_error <= 1e-12, (key, max_abs_error)
        torch.testing.assert_close(candidate, reference, rtol=0.0, atol=1e-12)

    saved_train = torch.load(tmp_path / "combined" / "teacher_dataset.pt")
    saved_validation = torch.load(
        tmp_path / "combined" / "validation_dataset.pt"
    )
    assert saved_train["endpoint"].shape[0] == train_size
    assert saved_validation["endpoint"].shape[0] == validation_size
    torch.testing.assert_close(
        saved_train["noise"], noises[:train_size], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        saved_validation["noise"],
        noises[train_size:],
        rtol=0.0,
        atol=0.0,
    )


def _fake_shard(start, end, *, total_size=4, train_size=3):
    size = end - start
    dataset = {
        "initial_point": torch.zeros(size, 3, dtype=DTYPE),
        "time": torch.zeros(size, dtype=DTYPE),
    }
    details = {
        "endpoint": torch.zeros(size, 3, dtype=DTYPE),
    }
    return {
        "format_version": 1,
        "teacher": "malliavin",
        "start": start,
        "end": end,
        "total_size": total_size,
        "train_size": train_size,
        "validation_size": total_size - train_size,
        "dataset_keys": list(dataset),
        "detail_keys": list(details),
        "dtype": "float64",
        "global_indices": torch.arange(start, end, dtype=torch.int64),
        "dataset": dataset,
        "teacher_details": details,
    }


def _save_fake_shards(tmp_path, first, second):
    paths = []
    for index, payload in enumerate((first, second)):
        path = tmp_path / f"fake_{index}.pt"
        torch.save(payload, path)
        paths.append(path)
    return paths


def test_combiner_rejects_missing_range(tmp_path):
    paths = _save_fake_shards(tmp_path, _fake_shard(0, 2), _fake_shard(3, 4))
    with pytest.raises(ValueError, match="missing teacher indices"):
        combine_teacher_shards(paths, output_dir=tmp_path / "out")


def test_combiner_rejects_overlapping_range(tmp_path):
    paths = _save_fake_shards(tmp_path, _fake_shard(0, 3), _fake_shard(2, 4))
    with pytest.raises(ValueError, match="overlapping teacher shard"):
        combine_teacher_shards(paths, output_dir=tmp_path / "out")


def test_combiner_rejects_out_of_order_indices(tmp_path):
    first = _fake_shard(0, 2)
    first["global_indices"] = torch.tensor([1, 0], dtype=torch.int64)
    paths = _save_fake_shards(tmp_path, first, _fake_shard(2, 4))
    with pytest.raises(ValueError, match="out of order"):
        combine_teacher_shards(paths, output_dir=tmp_path / "out")


@pytest.mark.parametrize("failure", ["shape", "dtype"])
def test_combiner_rejects_shape_and_dtype_mismatch(tmp_path, failure):
    second = copy.deepcopy(_fake_shard(2, 4))
    if failure == "shape":
        second["teacher_details"]["endpoint"] = torch.zeros(
            1, 3, dtype=DTYPE
        )
        match = "first dimension"
    else:
        second["dataset"]["time"] = second["dataset"]["time"].float()
        match = "dtype"
    paths = _save_fake_shards(tmp_path, _fake_shard(0, 2), second)
    with pytest.raises(ValueError, match=match):
        combine_teacher_shards(paths, output_dir=tmp_path / "out")
