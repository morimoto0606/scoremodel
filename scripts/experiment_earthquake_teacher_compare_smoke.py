#!/usr/bin/env python3
"""Earthquake smoke runner with fixed-condition teacher comparison on S2."""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Dict, Sequence, Tuple

import torch

from scoremodel_ext.manifold.earthquake_adapter import (
    load_earthquake_points,
    nearest_neighbor_geodesic_summary,
    s2_rbf_mmd,
)
from scoremodel_ext.manifold.experiment_s2_malliavin_teacher import (
    train_s2_marginal_score,
    train_s2_score_model,
)
from scoremodel_ext.malliavin.models import (
    MirafzaliSkorokhodNet,
    NormalizedSkorokhodModel,
)
from scoremodel_ext.manifold.s2_malliavin import (
    S2SkorokhodScoreModel,
    s2_discrete_malliavin_teacher,
    s2_grw_endpoint,
    s2_heat_kernel_score,
    s2_reverse_grw,
    s2_varadhan_score,
)


MAX_REVERSE_NOISE_STEPS = 1000

CURATED_LOWT_TIMES: Tuple[float, ...] = (
    0.005,
    0.010,
    0.020,
    0.035,
    0.050,
    0.075,
    0.100,
    0.150,
    0.200,
    0.250,
    0.300,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", choices=("heat", "varadhan", "malliavin"), default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("upstream/riemannian-score-sde/data/quakes_all.csv"),
    )
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--validation-size", type=int, default=128)
    parser.add_argument("--n-steps", type=int, default=8)
    parser.add_argument("--minimum-time", type=float, default=0.05)
    parser.add_argument("--maximum-time", type=float, default=0.3)
    parser.add_argument(
        "--time-sampling",
        choices=("uniform", "curated-lowt"),
        default="uniform",
    )
    parser.add_argument("--time-samples-path", type=Path, default=None)
    parser.add_argument("--validation-time-samples-path", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--num-frequencies", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--reverse-seed", type=int, default=0)
    parser.add_argument("--reverse-steps", type=int, default=32)
    parser.add_argument("--n-generated-samples", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--split-indices-path", type=Path, default=None)
    parser.add_argument("--terminal-samples-path", type=Path, default=None)
    parser.add_argument("--reverse-noise-path", type=Path, default=None)
    parser.add_argument("--skip-teacher-generation", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--covariance-regularization", type=float, default=1e-6)
    parser.add_argument("--heat-terms", type=int, default=80)
    parser.add_argument("--viz-output-dir", type=Path, default=None)
    parser.add_argument("--skip-viz", action="store_true")
    args = parser.parse_args()
    if args.skip_training and args.model_path is None:
        parser.error("--skip-training requires --model-path")
    if args.skip_teacher_generation and not args.skip_training:
        parser.error("--skip-teacher-generation requires --skip-training")
    if not args.skip_training and args.teacher is None:
        parser.error("--teacher is required unless --skip-training is used")
    return args


def to_dtype(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def resolve_device(name: str) -> str:
    if name == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return name


def normalize(points: torch.Tensor) -> torch.Tensor:
    return points / torch.linalg.vector_norm(points, dim=1, keepdim=True).clamp_min(1e-12)


def sample_curated_lowt_times(
    n_samples: int,
    *,
    dtype: torch.dtype,
    device: str,
    seed: int,
    curated_times: Sequence[float] = CURATED_LOWT_TIMES,
) -> torch.Tensor:
    """Sample the curated grid with counts differing by at most one."""

    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    candidates = torch.tensor(curated_times, dtype=dtype, device=device)
    base_count, remainder = divmod(n_samples, candidates.numel())
    counts = torch.full(
        (candidates.numel(),),
        base_count,
        dtype=torch.long,
        device=device,
    )
    counts[:remainder] += 1
    samples = torch.repeat_interleave(candidates, counts)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    permutation = torch.randperm(n_samples, generator=generator, device=device)
    return samples[permutation]


def create_time_samples(
    *,
    train_size: int,
    validation_size: int,
    time_sampling: str,
    minimum_time: float,
    maximum_time: float,
    dtype: torch.dtype,
    device: str,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create train/validation times while preserving legacy uniform draws."""

    if time_sampling == "uniform":
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        train_times = torch.empty(train_size, dtype=dtype, device=device).uniform_(
            minimum_time,
            maximum_time,
            generator=generator,
        )
        validation_times = torch.empty(
            validation_size,
            dtype=dtype,
            device=device,
        ).uniform_(
            minimum_time,
            maximum_time,
            generator=generator,
        )
        return train_times, validation_times

    if time_sampling != "curated-lowt":
        raise ValueError(f"unknown time_sampling: {time_sampling!r}")
    if minimum_time > CURATED_LOWT_TIMES[0] or maximum_time < CURATED_LOWT_TIMES[-1]:
        raise ValueError(
            "curated-lowt requires minimum_time <= 0.005 and maximum_time >= 0.3"
        )
    return (
        sample_curated_lowt_times(
            train_size,
            dtype=dtype,
            device=device,
            seed=seed,
        ),
        sample_curated_lowt_times(
            validation_size,
            dtype=dtype,
            device=device,
            seed=seed + 1,
        ),
    )


def _time_tensor_from_payload(payload, *, key: str) -> torch.Tensor:
    if isinstance(payload, torch.Tensor):
        return payload
    if isinstance(payload, dict) and key in payload:
        return payload[key]
    raise ValueError(f"time sample artifact does not contain {key!r}")


def load_time_samples(
    *,
    train_path: Path,
    validation_path: Path | None,
    dtype: torch.dtype,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load new split artifacts or the legacy combined dictionary artifact."""

    train_payload = torch.load(train_path, map_location="cpu")
    train_times = _time_tensor_from_payload(train_payload, key="train_times")
    if validation_path is None:
        validation_times = _time_tensor_from_payload(
            train_payload,
            key="validation_times",
        )
    else:
        validation_payload = torch.load(validation_path, map_location="cpu")
        validation_times = _time_tensor_from_payload(
            validation_payload,
            key="validation_times",
        )
    return (
        train_times.detach().to(device=device, dtype=dtype),
        validation_times.detach().to(device=device, dtype=dtype),
    )


def validate_time_samples(
    train_times: torch.Tensor,
    validation_times: torch.Tensor,
    *,
    train_size: int,
    validation_size: int,
    time_sampling: str,
) -> None:
    if train_times.shape != (train_size,):
        raise ValueError(f"train time samples must have shape {(train_size,)}")
    if validation_times.shape != (validation_size,):
        raise ValueError(
            f"validation time samples must have shape {(validation_size,)}"
        )
    if not bool(torch.isfinite(train_times).all()) or not bool(
        torch.isfinite(validation_times).all()
    ):
        raise ValueError("time samples must be finite")
    if time_sampling == "curated-lowt":
        candidates = torch.tensor(
            CURATED_LOWT_TIMES,
            dtype=train_times.dtype,
            device=train_times.device,
        )
        for name, values in (
            ("train", train_times),
            ("validation", validation_times),
        ):
            is_curated = torch.isclose(
                values[:, None],
                candidates[None, :],
                rtol=0.0,
                atol=1e-12,
            ).any(dim=1)
            if not bool(is_curated.all()):
                raise ValueError(f"{name} time samples contain non-curated values")


def save_time_samples(
    train_times: torch.Tensor,
    validation_times: torch.Tensor,
    *,
    train_path: Path,
    validation_path: Path,
) -> None:
    torch.save(train_times.detach().cpu(), train_path)
    torch.save(validation_times.detach().cpu(), validation_path)


def compute_split_indices(
    n_total: int,
    train_size: int,
    validation_size: int,
    *,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if train_size < 1 or validation_size < 1:
        raise ValueError("train_size and validation_size must be positive")
    if train_size + validation_size > n_total:
        raise ValueError("train_size + validation_size exceeds available earthquake samples")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(n_total, generator=generator)
    train_index = permutation[:train_size]
    validation_index = permutation[train_size : train_size + validation_size]
    return train_index, validation_index


def maybe_load_or_create_split_indices(
    *,
    split_indices_path: Path | None,
    output_dir: Path,
    n_total: int,
    train_size: int,
    validation_size: int,
    split_seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, Path, Path]:
    train_path = output_dir / "train_indices.pt"
    validation_path = output_dir / "validation_indices.pt"

    if split_indices_path is not None:
        root = split_indices_path
        if root.is_file():
            payload = torch.load(root, map_location="cpu")
            train_idx = payload["train_indices"].long()
            validation_idx = payload["validation_indices"].long()
            torch.save(train_idx, train_path)
            torch.save(validation_idx, validation_path)
            return train_idx, validation_idx, train_path, validation_path
        root.mkdir(parents=True, exist_ok=True)
        root_train = root / "train_indices.pt"
        root_validation = root / "validation_indices.pt"
        if root_train.exists() and root_validation.exists():
            train_idx = torch.load(root_train, map_location="cpu").long()
            validation_idx = torch.load(root_validation, map_location="cpu").long()
            torch.save(train_idx, train_path)
            torch.save(validation_idx, validation_path)
            return train_idx, validation_idx, train_path, validation_path

    train_idx, validation_idx = compute_split_indices(
        n_total,
        train_size,
        validation_size,
        seed=split_seed,
    )
    torch.save(train_idx, train_path)
    torch.save(validation_idx, validation_path)
    if split_indices_path is not None:
        root = split_indices_path
        root.mkdir(parents=True, exist_ok=True)
        torch.save(train_idx, root / "train_indices.pt")
        torch.save(validation_idx, root / "validation_indices.pt")
    return train_idx, validation_idx, train_path, validation_path


def maybe_load_or_create_terminal_samples(
    *,
    path: Path | None,
    output_path: Path,
    n_generated_samples: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
) -> torch.Tensor:
    if path is not None and path.exists():
        tensor = torch.load(path, map_location="cpu")
        if isinstance(tensor, dict):
            tensor = tensor["terminal_samples"]
        tensor = tensor.to(device=device, dtype=dtype)
        torch.save(tensor.detach().cpu(), output_path)
        return tensor

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    samples = torch.randn(n_generated_samples, 3, generator=generator, dtype=dtype, device=device)
    samples = normalize(samples)
    torch.save(samples.detach().cpu(), output_path)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(samples.detach().cpu(), path)
    return samples


def maybe_load_or_create_reverse_noise(
    *,
    path: Path | None,
    output_path: Path,
    reverse_steps: int,
    n_generated_samples: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
) -> torch.Tensor:
    if path is not None and path.exists():
        tensor = torch.load(path, map_location="cpu")
        if isinstance(tensor, dict):
            tensor = tensor["reverse_noise"]
        tensor = tensor.to(device=device, dtype=dtype)
        torch.save(tensor.detach().cpu(), output_path)
        return tensor

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    noise = torch.randn(reverse_steps, n_generated_samples, 3, generator=generator, dtype=dtype, device=device)
    torch.save(noise.detach().cpu(), output_path)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(noise.detach().cpu(), path)
    return noise


def load_run_config_for_model(model_path: Path) -> tuple[Path, Dict[str, object]]:
    """Load the configuration adjacent to a saved model checkpoint."""

    resolved_model_path = model_path.expanduser().resolve()
    if not resolved_model_path.is_file():
        raise FileNotFoundError(f"missing saved model: {resolved_model_path}")
    config_path = resolved_model_path.parent / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"missing run_config.json next to saved model: {config_path}"
        )
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return resolved_model_path, config


def build_model_from_run_config(
    model_path: Path,
    run_config: Dict[str, object],
    *,
    device: str,
):
    """Reconstruct the exact saved network structure and load its weights."""

    required = ("teacher", "hidden", "n_blocks", "num_frequencies", "dtype")
    missing = [key for key in required if key not in run_config]
    if missing:
        raise ValueError(f"run_config.json is missing model fields: {missing}")

    dtype = to_dtype(str(run_config["dtype"]))
    network = MirafzaliSkorokhodNet(
        x_dim=3,
        out_dim=3,
        hidden=int(run_config["hidden"]),
        n_blocks=int(run_config["n_blocks"]),
        num_frequencies=int(run_config["num_frequencies"]),
    ).to(device=device, dtype=dtype)
    zeros_vector = torch.zeros(1, 3, dtype=dtype, device=device)
    ones_vector = torch.ones(1, 3, dtype=dtype, device=device)
    zeros_time = torch.zeros(1, 1, dtype=dtype, device=device)
    ones_time = torch.ones(1, 1, dtype=dtype, device=device)
    normalized_model = NormalizedSkorokhodModel(
        network,
        zeros_vector,
        ones_vector,
        zeros_time,
        ones_time,
        zeros_vector,
        ones_vector,
    ).to(device=device, dtype=dtype)
    if run_config["teacher"] == "malliavin":
        model = S2SkorokhodScoreModel(normalized_model).to(device=device, dtype=dtype)
    else:
        model = normalized_model

    checkpoint = torch.load(model_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"saved model has no state_dict: {model_path}")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


def maybe_load_or_create_shared_reverse_noise(
    *,
    path: Path,
    output_path: Path,
    reverse_steps: int,
    n_generated_samples: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
) -> torch.Tensor:
    """Persist fine noise and approximately couple coarse reverse-step runs."""

    if not 1 <= reverse_steps <= MAX_REVERSE_NOISE_STEPS:
        raise ValueError(
            f"reverse_steps must be between 1 and {MAX_REVERSE_NOISE_STEPS}"
        )
    if path.exists():
        payload = torch.load(path, map_location="cpu")
        pool = payload["reverse_noise"] if isinstance(payload, dict) else payload
        expected_shape = (MAX_REVERSE_NOISE_STEPS, n_generated_samples, 3)
        if tuple(pool.shape) != expected_shape:
            raise ValueError(
                f"shared reverse noise must have shape {expected_shape}, "
                f"got {tuple(pool.shape)} from {path}"
            )
        pool = pool.to(device=device, dtype=dtype)
    else:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        pool = torch.randn(
            MAX_REVERSE_NOISE_STEPS,
            n_generated_samples,
            3,
            generator=generator,
            dtype=dtype,
            device=device,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(pool.detach().cpu(), path)

    result = aggregate_reverse_noise_pool(pool, reverse_steps=reverse_steps)
    if output_path.resolve() != path.resolve():
        torch.save(result.detach().cpu(), output_path)
    return result


def aggregate_reverse_noise_pool(
    pool: torch.Tensor,
    *,
    reverse_steps: int,
) -> torch.Tensor:
    """Convert fine standard normals to coarse ones via path interpolation.

    The cumulative sum defines a Brownian path on the common fine grid.  The
    path is linearly interpolated at coarse-grid times, differenced, and divided
    by the coarse ``sqrt(dt)``.  This approximately couples non-divisor step
    counts through a common 1000-step fine noise pool.  When ``reverse_steps``
    divides the pool length, interpolation lands on fine-grid boundaries and
    the result is the exact normalized sum of each fine-increment block.
    """

    if pool.ndim != 3 or pool.shape[2] != 3:
        raise ValueError("reverse noise pool must have shape [steps, batch, 3]")
    fine_steps = int(pool.shape[0])
    if not 1 <= reverse_steps <= fine_steps:
        raise ValueError(f"reverse_steps must be between 1 and {fine_steps}")

    cumulative = torch.cat(
        (
            torch.zeros(
                1,
                pool.shape[1],
                pool.shape[2],
                dtype=pool.dtype,
                device=pool.device,
            ),
            torch.cumsum(pool, dim=0),
        ),
        dim=0,
    )
    coarse_positions = torch.arange(
        reverse_steps + 1,
        dtype=torch.float64,
        device=pool.device,
    ) * (fine_steps / reverse_steps)
    lower = torch.floor(coarse_positions).to(torch.long)
    upper = torch.ceil(coarse_positions).to(torch.long).clamp_max(fine_steps)
    fraction = (coarse_positions - lower).to(dtype=pool.dtype)
    interpolated = (
        cumulative[lower] * (1.0 - fraction[:, None, None])
        + cumulative[upper] * fraction[:, None, None]
    )
    path_increments = interpolated[1:] - interpolated[:-1]
    return path_increments * (reverse_steps / fine_steps) ** 0.5


def build_teacher_dataset(
    *,
    initial_points: torch.Tensor,
    times: torch.Tensor,
    noises: torch.Tensor,
    teacher: str,
    covariance_regularization: float,
    heat_terms: int,
) -> Dict[str, torch.Tensor]:
    endpoints = []
    score_target = []
    skorokhod = []

    for initial_point, terminal_time, noise in zip(initial_points, times, noises):
        terminal_time_float = float(terminal_time.detach().cpu())
        if teacher == "malliavin":
            teacher_state = s2_discrete_malliavin_teacher(
                initial_point,
                noise,
                terminal_time_float,
                covariance_regularization=covariance_regularization,
                vectorize_jacobian=True,
            )
            endpoints.append(teacher_state.endpoint)
            score_target.append(teacher_state.score_weight)
            skorokhod.append(teacher_state.skorokhod)
        elif teacher == "heat":
            endpoint = s2_grw_endpoint(initial_point, noise, terminal_time_float)
            endpoints.append(endpoint)
            score_target.append(
                s2_heat_kernel_score(
                    initial_point,
                    endpoint,
                    terminal_time_float,
                    n_terms=heat_terms,
                )
            )
        else:
            endpoint = s2_grw_endpoint(initial_point, noise, terminal_time_float)
            endpoints.append(endpoint)
            score_target.append(s2_varadhan_score(initial_point, endpoint, terminal_time_float))

    dataset: Dict[str, torch.Tensor] = {
        "initial_point": initial_points,
        "time": times,
        "noise": noises,
        "endpoint": torch.stack(endpoints, dim=0),
        "score_target": torch.stack(score_target, dim=0),
    }
    if teacher == "malliavin":
        dataset["skorokhod"] = torch.stack(skorokhod, dim=0)
    return {
        key: value.detach() if isinstance(value, torch.Tensor) else value
        for key, value in dataset.items()
    }


def evaluate_dataset_loss(
    model,
    dataset: Dict[str, torch.Tensor],
    *,
    teacher: str,
) -> float:
    with torch.no_grad():
        if teacher == "malliavin":
            prediction = model.skorokhod_network(dataset["time"], dataset["endpoint"])
            target = dataset["skorokhod"]
        else:
            prediction = model(dataset["time"], dataset["endpoint"])
            target = dataset["score_target"]
        value = torch.mean((prediction - target) ** 2)
    return float(value)


def build_score_fn(model):
    def _score_fn(time_batch: torch.Tensor, x_batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model(time_batch, x_batch)

    return _score_fn


def generated_norm_error(samples: torch.Tensor) -> float:
    norm = torch.linalg.vector_norm(samples, dim=1)
    return float(torch.mean(torch.abs(norm - 1.0)))


def finite_rate(samples: torch.Tensor) -> float:
    return float(torch.isfinite(samples).all(dim=1).double().mean())


def _nearest_curated_time_index(times: torch.Tensor) -> torch.Tensor:
    candidates = torch.tensor(
        CURATED_LOWT_TIMES,
        dtype=times.dtype,
        device=times.device,
    )
    return torch.argmin(torch.abs(times[:, None] - candidates[None, :]), dim=1)


def time_histogram(times: torch.Tensor) -> Dict[str, int]:
    """Count samples in bins centered at the shared curated time grid."""

    assignment = _nearest_curated_time_index(times)
    return {
        f"{candidate:.3f}": int((assignment == index).sum().detach().cpu())
        for index, candidate in enumerate(CURATED_LOWT_TIMES)
    }


def samples_per_curated_time(times: torch.Tensor) -> list[int]:
    """Return exact curated counts aligned with ``CURATED_LOWT_TIMES``."""

    return [
        int(
            torch.isclose(
                times,
                torch.tensor(candidate, dtype=times.dtype, device=times.device),
                rtol=0.0,
                atol=1e-12,
            )
            .sum()
            .detach()
            .cpu()
        )
        for candidate in CURATED_LOWT_TIMES
    ]


def time_target_diagnostics(
    dataset: Dict[str, torch.Tensor],
    *,
    target_key: str,
) -> list[dict]:
    """Summarize target magnitude and time-local dispersion on shared bins.

    ``time_bin_target_dispersion`` is the RMS deviation from the mean target
    within a time bin divided by target RMS in that bin.
    """

    times = dataset["time"].detach()
    targets = dataset[target_key].detach()
    assignment = _nearest_curated_time_index(times)
    rows = []
    for index, candidate in enumerate(CURATED_LOWT_TIMES):
        mask = assignment == index
        count = int(mask.sum().detach().cpu())
        if count == 0:
            rows.append(
                {
                    "time": candidate,
                    "count": 0,
                    "target_norm_mean": None,
                    "target_norm_std": None,
                    "time_bin_target_dispersion": None,
                }
            )
            continue
        selected = targets[mask]
        norms = torch.linalg.vector_norm(selected, dim=1)
        centered = selected - selected.mean(dim=0, keepdim=True)
        residual_rms = torch.sqrt(torch.mean(torch.sum(centered**2, dim=1)))
        target_rms = torch.sqrt(torch.mean(torch.sum(selected**2, dim=1)))
        rows.append(
            {
                "time": candidate,
                "count": count,
                "target_norm_mean": float(norms.mean().detach().cpu()),
                "target_norm_std": float(
                    norms.std(unbiased=False).detach().cpu()
                ),
                "time_bin_target_dispersion": float(
                    (residual_rms / target_rms.clamp_min(1e-12)).detach().cpu()
                ),
            }
        )
    return rows


def diagnostic_target_key_for_teacher(teacher: str) -> str:
    return "skorokhod" if teacher == "malliavin" else "score_target"


def _load_saved_evaluation_points(
    model_dir: Path,
    run_config: Dict[str, object],
    *,
    dtype: torch.dtype,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load observed train/validation points without rebuilding teacher data."""

    target_path = model_dir / "target_samples.pt"
    if not target_path.is_file():
        raise FileNotFoundError(
            f"saved evaluation target is required in reuse mode: {target_path}"
        )
    validation_points = torch.load(target_path, map_location="cpu").to(
        device=device,
        dtype=dtype,
    )

    data_path = Path(str(run_config["data_path"]))
    if not data_path.is_absolute() and not data_path.is_file():
        data_path = Path(__file__).resolve().parents[1] / data_path
    train_indices_path = model_dir / "train_indices.pt"
    if data_path.is_file() and train_indices_path.is_file():
        all_points = load_earthquake_points(data_path, dtype=dtype, device=device)
        train_indices = torch.load(train_indices_path, map_location="cpu").long()
        return normalize(all_points[train_indices.to(device)]), normalize(validation_points)

    # Older run directories may only retain the initial points inside the
    # serialized training artifact.  Loading it is a compatibility fallback;
    # no teacher targets are generated or evaluated in this mode.
    dataset_path = model_dir / "teacher_dataset.pt"
    if dataset_path.is_file():
        payload = torch.load(dataset_path, map_location="cpu")
        if isinstance(payload, dict) and "initial_point" in payload:
            train_points = payload["initial_point"].to(device=device, dtype=dtype)
            return normalize(train_points), normalize(validation_points)

    warnings.warn(
        "saved training points are unavailable; the density target uses the "
        "saved validation points only",
        RuntimeWarning,
        stacklevel=1,
    )
    return validation_points[:0], normalize(validation_points)


def run_saved_model_evaluation(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    log,
) -> None:
    """Run only reverse sampling and sample-based evaluation."""

    if args.model_path is None:
        raise ValueError("--skip-training requires --model-path")
    model_path, source_config = load_run_config_for_model(args.model_path)
    source_model_dir = model_path.parent

    required_config = (
        "teacher",
        "dtype",
        "minimum_time",
        "maximum_time",
        "n_generated_samples",
        "seed",
        "reverse_seed",
    )
    missing = [key for key in required_config if key not in source_config]
    if missing:
        raise ValueError(f"run_config.json is missing evaluation fields: {missing}")

    teacher = str(source_config["teacher"])
    dtype_name = str(source_config["dtype"])
    dtype = to_dtype(dtype_name)
    device = resolve_device(args.device)
    n_generated_samples = int(source_config["n_generated_samples"])
    reverse_steps = int(args.reverse_steps)

    log(f"loading saved {teacher} model from {model_path}")
    model = build_model_from_run_config(model_path, source_config, device=device)
    observed_train, observed_validation = _load_saved_evaluation_points(
        source_model_dir,
        source_config,
        dtype=dtype,
        device=device,
    )

    terminal_path = (
        args.terminal_samples_path.expanduser().resolve()
        if args.terminal_samples_path is not None
        else source_model_dir / "terminal_samples.pt"
    )
    terminal_samples = maybe_load_or_create_terminal_samples(
        path=terminal_path,
        output_path=output_dir / "terminal_samples.pt",
        n_generated_samples=n_generated_samples,
        dtype=dtype,
        device=device,
        seed=int(source_config["seed"]) + 101,
    )
    expected_terminal_shape = (n_generated_samples, 3)
    if tuple(terminal_samples.shape) != expected_terminal_shape:
        raise ValueError(
            f"terminal samples must have shape {expected_terminal_shape}, "
            f"got {tuple(terminal_samples.shape)}"
        )

    shared_noise_path = (
        args.reverse_noise_path.expanduser().resolve()
        if args.reverse_noise_path is not None
        else source_model_dir / "reverse_noise_1000.pt"
    )
    reverse_noise = maybe_load_or_create_shared_reverse_noise(
        path=shared_noise_path,
        output_path=output_dir / "reverse_noise.pt",
        reverse_steps=reverse_steps,
        n_generated_samples=n_generated_samples,
        dtype=dtype,
        device=device,
        seed=int(source_config["reverse_seed"]),
    )

    reverse_started = time.perf_counter()
    generated = s2_reverse_grw(
        terminal_samples,
        build_score_fn(model),
        terminal_time=float(source_config["maximum_time"]),
        n_steps=reverse_steps,
        standard_noise=reverse_noise,
        minimum_forward_time=float(source_config["minimum_time"]),
    )
    reverse_sampling_seconds = time.perf_counter() - reverse_started
    generated_cpu = generated.detach().cpu()
    target_cpu = observed_validation.detach().cpu()
    torch.save(generated_cpu, output_dir / "generated_samples.pt")
    torch.save(target_cpu, output_dir / "target_samples.pt")

    evaluation_seed = int(source_config["seed"])
    mmd_value = s2_rbf_mmd(generated_cpu, target_cpu, seed=evaluation_seed)
    generated_to_target = nearest_neighbor_geodesic_summary(
        generated_cpu,
        target_cpu,
        seed=evaluation_seed,
    )
    target_to_generated = nearest_neighbor_geodesic_summary(
        target_cpu,
        generated_cpu,
        seed=evaluation_seed,
    )
    sample_finite_rate = finite_rate(generated_cpu)
    metrics = {
        "teacher": teacher,
        "evaluation_only": True,
        "reverse_steps": reverse_steps,
        "reverse_sampling_seconds": reverse_sampling_seconds,
        "s2_rbf_mmd": mmd_value,
        "generated_to_target_nearest_neighbor_geodesic": generated_to_target,
        "target_to_generated_nearest_neighbor_geodesic": target_to_generated,
        "nearest_neighbor_geodesic_mean": generated_to_target["mean"],
        "nearest_neighbor_geodesic_median": generated_to_target["median"],
        "nearest_neighbor_geodesic_max": generated_to_target["max"],
        "generated_to_target_nearest_neighbor_geodesic_mean": generated_to_target["mean"],
        "generated_to_target_nearest_neighbor_geodesic_median": generated_to_target["median"],
        "generated_to_target_nearest_neighbor_geodesic_max": generated_to_target["max"],
        "target_to_generated_nearest_neighbor_geodesic_mean": target_to_generated["mean"],
        "target_to_generated_nearest_neighbor_geodesic_median": target_to_generated["median"],
        "target_to_generated_nearest_neighbor_geodesic_max": target_to_generated["max"],
        "generated_sample_norm_error": generated_norm_error(generated_cpu),
        "nan_rate": 1.0 - sample_finite_rate,
        "generated_all_finite": bool(sample_finite_rate == 1.0),
        "generated_sample_count": int(generated_cpu.shape[0]),
        "device": device,
        "dtype": dtype_name,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    evaluation_config = dict(source_config)
    evaluation_config.update(
        {
            "output_dir": str(output_dir),
            "model_path": str(model_path),
            "source_run_config": str(source_model_dir / "run_config.json"),
            "skip_teacher_generation": True,
            "skip_training": True,
            "reverse_steps": reverse_steps,
            "terminal_samples_path": str(terminal_path),
            "reverse_noise_path": str(shared_noise_path),
            "reverse_noise_pool_steps": MAX_REVERSE_NOISE_STEPS,
            "reverse_noise_coupling": (
                "linear_interpolation_of_cumulative_fine_brownian_path"
            ),
            "reverse_noise_coupling_exact": (
                MAX_REVERSE_NOISE_STEPS % reverse_steps == 0
            ),
            "resolved_device": device,
        }
    )
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(evaluation_config, handle, indent=2, default=str)

    from scoremodel_ext.manifold.earthquake_smoke_viz import (
        generate_earthquake_density_plots,
    )

    viz_dir = (
        args.viz_output_dir.expanduser().resolve()
        if args.viz_output_dir is not None
        else output_dir
    )
    generate_earthquake_density_plots(
        observed_train_points=observed_train.detach().cpu(),
        observed_validation_points=target_cpu,
        generated_by_teacher={teacher: generated_cpu},
        output_dir=viz_dir,
    )
    log(
        f"evaluation-only reverse_steps={reverse_steps} mmd={mmd_value:.6e} "
        f"generated_to_target_nn={generated_to_target['mean']:.6e} "
        f"target_to_generated_nn={target_to_generated['mean']:.6e}"
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_log_path = output_dir / "run.log"

    def log(message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line)
        with run_log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    if args.skip_teacher_generation and not args.skip_training:
        raise ValueError("--skip-teacher-generation requires --skip-training")
    if args.skip_training:
        run_saved_model_evaluation(args, output_dir=output_dir, log=log)
        return
    if args.teacher is None:
        raise ValueError("--teacher is required unless --skip-training is used")

    dtype = to_dtype(args.dtype)
    device = resolve_device(args.device)

    if args.teacher != "heat":
        heat_dir = output_dir.parent / "heat"
        if args.split_indices_path is None and (heat_dir / "train_indices.pt").exists() and (heat_dir / "validation_indices.pt").exists():
            args.split_indices_path = heat_dir
        if args.terminal_samples_path is None and (heat_dir / "terminal_samples.pt").exists():
            args.terminal_samples_path = heat_dir / "terminal_samples.pt"
        if args.reverse_noise_path is None and (heat_dir / "reverse_noise.pt").exists():
            args.reverse_noise_path = heat_dir / "reverse_noise.pt"
        if (
            args.time_samples_path is None
            and args.validation_time_samples_path is None
            and (heat_dir / "time_samples.pt").exists()
        ):
            args.time_samples_path = heat_dir / "time_samples.pt"
            validation_time_path = heat_dir / "validation_time_samples.pt"
            if validation_time_path.exists():
                args.validation_time_samples_path = validation_time_path

    log(f"loading earthquake points from {args.data_path}")
    points = load_earthquake_points(args.data_path, dtype=dtype, device=device)

    train_idx, validation_idx, train_idx_path, validation_idx_path = maybe_load_or_create_split_indices(
        split_indices_path=args.split_indices_path,
        output_dir=output_dir,
        n_total=points.shape[0],
        train_size=args.train_size,
        validation_size=args.validation_size,
        split_seed=args.split_seed,
    )
    log(f"split indices ready: {train_idx_path.name}, {validation_idx_path.name}")

    train_initial = normalize(points[train_idx.to(device=device)])
    validation_initial = normalize(points[validation_idx.to(device=device)])

    time_samples_path = output_dir / "time_samples.pt"
    validation_time_samples_path = output_dir / "validation_time_samples.pt"
    noise_samples_path = output_dir / "teacher_noises.pt"
    if args.teacher != "heat":
        sibling = output_dir.parent / "heat"
        sibling_noise = sibling / "teacher_noises.pt"
    else:
        sibling_noise = None

    if (
        args.time_samples_path is None
        and args.validation_time_samples_path is not None
    ):
        raise ValueError(
            "--validation-time-samples-path requires --time-samples-path"
        )
    if args.time_samples_path is not None:
        if not args.time_samples_path.exists():
            raise FileNotFoundError(
                f"missing train time samples: {args.time_samples_path}"
            )
        if (
            args.validation_time_samples_path is not None
            and not args.validation_time_samples_path.exists()
        ):
            raise FileNotFoundError(
                "missing validation time samples: "
                f"{args.validation_time_samples_path}"
            )
        train_times, validation_times = load_time_samples(
            train_path=args.time_samples_path,
            validation_path=args.validation_time_samples_path,
            dtype=dtype,
            device=device,
        )
    else:
        train_times, validation_times = create_time_samples(
            train_size=args.train_size,
            validation_size=args.validation_size,
            time_sampling=args.time_sampling,
            minimum_time=args.minimum_time,
            maximum_time=args.maximum_time,
            dtype=dtype,
            device=device,
            seed=args.seed,
        )
    validate_time_samples(
        train_times,
        validation_times,
        train_size=args.train_size,
        validation_size=args.validation_size,
        time_sampling=args.time_sampling,
    )
    save_time_samples(
        train_times,
        validation_times,
        train_path=time_samples_path,
        validation_path=validation_time_samples_path,
    )
    time_histograms = {
        "train": time_histogram(train_times),
        "validation": time_histogram(validation_times),
    }
    samples_per_time = {
        "train": (
            samples_per_curated_time(train_times)
            if args.time_sampling == "curated-lowt"
            else None
        ),
        "validation": (
            samples_per_curated_time(validation_times)
            if args.time_sampling == "curated-lowt"
            else None
        ),
    }

    run_config = vars(args).copy()
    run_config.update(
        {
            "resolved_device": device,
            "curated_times": list(CURATED_LOWT_TIMES),
            "time_histogram": time_histograms,
            "samples_per_time": samples_per_time,
        }
    )
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2, default=str)
    log(
        f"time samples ready: mode={args.time_sampling} "
        f"train={time_samples_path.name} "
        f"validation={validation_time_samples_path.name}"
    )

    if sibling_noise is not None and sibling_noise.exists():
        payload = torch.load(sibling_noise, map_location="cpu")
        train_noises = payload["train_noises"].to(device=device, dtype=dtype)
        validation_noises = payload["validation_noises"].to(device=device, dtype=dtype)
    else:
        noise_generator = torch.Generator(device=device)
        noise_generator.manual_seed(args.seed + 11)
        train_noises = torch.randn(args.train_size, args.n_steps, 3, generator=noise_generator, dtype=dtype, device=device)
        validation_noises = torch.randn(
            args.validation_size,
            args.n_steps,
            3,
            generator=noise_generator,
            dtype=dtype,
            device=device,
        )
    torch.save(
        {
            "train_noises": train_noises.detach().cpu(),
            "validation_noises": validation_noises.detach().cpu(),
        },
        noise_samples_path,
    )

    teacher_started = time.perf_counter()
    train_dataset = build_teacher_dataset(
        initial_points=train_initial,
        times=train_times,
        noises=train_noises,
        teacher=args.teacher,
        covariance_regularization=args.covariance_regularization,
        heat_terms=args.heat_terms,
    )
    validation_dataset = build_teacher_dataset(
        initial_points=validation_initial,
        times=validation_times,
        noises=validation_noises,
        teacher=args.teacher,
        covariance_regularization=args.covariance_regularization,
        heat_terms=args.heat_terms,
    )
    teacher_generation_seconds = time.perf_counter() - teacher_started

    train_started = time.perf_counter()
    if args.teacher == "malliavin":
        model, history = train_s2_marginal_score(
            train_dataset,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            hidden=args.hidden,
            n_blocks=args.n_blocks,
            num_frequencies=args.num_frequencies,
            device=device,
            return_history=True,
        )
        training_path = "marginal_skorokhod"
    else:
        model, history = train_s2_score_model(
            train_dataset,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            hidden=args.hidden,
            n_blocks=args.n_blocks,
            num_frequencies=args.num_frequencies,
            device=device,
            return_history=True,
        )
        training_path = "direct_score"
    training_seconds = time.perf_counter() - train_started

    initial_train_loss = float(history.get("initial_train_loss", float("nan")))
    final_train_loss = float(history.get("final_train_loss", float("nan")))
    best_train_loss = float(history.get("best_train_loss", float("nan")))

    validation_loss = evaluate_dataset_loss(model, validation_dataset, teacher=args.teacher)

    terminal_samples = maybe_load_or_create_terminal_samples(
        path=args.terminal_samples_path,
        output_path=output_dir / "terminal_samples.pt",
        n_generated_samples=args.n_generated_samples,
        dtype=dtype,
        device=device,
        seed=args.seed + 101,
    )
    reverse_noise = maybe_load_or_create_reverse_noise(
        path=args.reverse_noise_path,
        output_path=output_dir / "reverse_noise.pt",
        reverse_steps=args.reverse_steps,
        n_generated_samples=args.n_generated_samples,
        dtype=dtype,
        device=device,
        seed=args.reverse_seed,
    )

    reverse_started = time.perf_counter()
    generated = s2_reverse_grw(
        terminal_samples,
        build_score_fn(model),
        terminal_time=args.maximum_time,
        n_steps=args.reverse_steps,
        standard_noise=reverse_noise,
        minimum_forward_time=args.minimum_time,
    )
    reverse_sampling_seconds = time.perf_counter() - reverse_started

    generated_cpu = generated.detach().cpu()
    train_dataset_cpu = {key: value.detach().cpu() for key, value in train_dataset.items()}
    validation_dataset_cpu = {key: value.detach().cpu() for key, value in validation_dataset.items()}

    torch.save(
        {
            "teacher": args.teacher,
            "training_path": training_path,
            "state_dict": model.state_dict(),
            "hidden": args.hidden,
            "n_blocks": args.n_blocks,
            "num_frequencies": args.num_frequencies,
            "dtype": args.dtype,
        },
        output_dir / "model.pt",
    )
    with (output_dir / "training_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    torch.save(generated_cpu, output_dir / "generated_samples.pt")
    torch.save(validation_initial.detach().cpu(), output_dir / "target_samples.pt")
    torch.save(train_dataset_cpu, output_dir / "teacher_dataset.pt")
    torch.save(validation_dataset_cpu, output_dir / "validation_dataset.pt")

    mmd_value = s2_rbf_mmd(generated_cpu, validation_initial.detach().cpu(), seed=args.seed)
    geodesic = nearest_neighbor_geodesic_summary(generated_cpu, validation_initial.detach().cpu(), seed=args.seed)
    norm_error = generated_norm_error(generated_cpu)
    sample_finite_rate = finite_rate(generated_cpu)
    diagnostic_target_key = diagnostic_target_key_for_teacher(args.teacher)
    time_diagnostics = {
        "train": time_target_diagnostics(
            train_dataset,
            target_key=diagnostic_target_key,
        ),
        "validation": time_target_diagnostics(
            validation_dataset,
            target_key=diagnostic_target_key,
        ),
    }

    metrics = {
        "teacher": args.teacher,
        "training_path": training_path,
        "initial_train_loss": initial_train_loss,
        "final_train_loss": final_train_loss,
        "best_train_loss": best_train_loss,
        "validation_loss": validation_loss,
        "teacher_generation_seconds": teacher_generation_seconds,
        "training_seconds": training_seconds,
        "reverse_sampling_seconds": reverse_sampling_seconds,
        "s2_rbf_mmd": mmd_value,
        "nearest_neighbor_geodesic_mean": geodesic["mean"],
        "nearest_neighbor_geodesic_median": geodesic["median"],
        "nearest_neighbor_geodesic_max": geodesic["max"],
        "generated_sample_norm_error": norm_error,
        "nan_rate": 1.0 - sample_finite_rate,
        "generated_all_finite": bool(sample_finite_rate == 1.0),
        "generated_sample_count": int(generated_cpu.shape[0]),
        "time_sampling": args.time_sampling,
        "curated_times": list(CURATED_LOWT_TIMES),
        "time_histogram": time_histograms,
        "samples_per_time": samples_per_time,
        "time_diagnostics": time_diagnostics,
        "diagnostic_target_key": diagnostic_target_key,
        "time_bin_target_dispersion_definition": (
            f"Within each shared time bin: RMS({diagnostic_target_key} - "
            f"bin mean) / RMS({diagnostic_target_key})."
        ),
        "device": device,
        "dtype": args.dtype,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    log(f"teacher={args.teacher} training_path={training_path}")
    log(f"final_train_loss={final_train_loss:.6e} validation_loss={validation_loss:.6e}")
    log(f"mmd={mmd_value:.6e} geodesic_mean={geodesic['mean']:.6e} norm_error={norm_error:.6e}")

    if not args.skip_viz or args.teacher == "malliavin":
        from scoremodel_ext.manifold.earthquake_smoke_viz import (
            generate_earthquake_density_plots,
            generate_earthquake_smoke_plots,
        )

        if args.viz_output_dir is not None:
            viz_dir = args.viz_output_dir.resolve()
        elif args.teacher == "malliavin":
            viz_dir = output_dir.parent
        else:
            viz_dir = output_dir

        teacher_generated: Dict[str, torch.Tensor] = {}
        for teacher_name in ("heat", "varadhan", "malliavin"):
            generated_path = output_dir.parent / teacher_name / "generated_samples.pt"
            if generated_path.exists():
                teacher_generated[teacher_name] = torch.load(
                    generated_path,
                    map_location="cpu",
                )
            elif teacher_name == args.teacher:
                teacher_generated[teacher_name] = generated_cpu
            elif args.teacher == "malliavin":
                warnings.warn(
                    f"missing {teacher_name} density artifact: {generated_path}; "
                    "generating comparison with available teachers",
                    RuntimeWarning,
                    stacklevel=1,
                )

        teacher_history: Dict[str, Dict[str, list[float]]] = {
            args.teacher: {
                "epochs": [int(x) for x in history.get("epochs", [])],
                "train_loss": [float(x) for x in history.get("train_loss", [])],
            }
        }
        teacher_time_diagnostics: Dict[str, list[dict]] = {
            args.teacher: time_diagnostics["validation"]
        }

        for other in ("heat", "varadhan", "malliavin"):
            if other == args.teacher:
                continue
            other_run = output_dir.parent / other
            other_history_path = other_run / "training_history.json"
            other_metrics_path = other_run / "metrics.json"
            if other_history_path.exists():
                with other_history_path.open("r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                teacher_history[other] = {
                    "epochs": [int(x) for x in loaded.get("epochs", [])],
                    "train_loss": [float(x) for x in loaded.get("train_loss", [])],
                }
                if other_metrics_path.exists():
                    with other_metrics_path.open("r", encoding="utf-8") as handle:
                        other_metrics = json.load(handle)
                    validation_diagnostics = other_metrics.get(
                        "time_diagnostics", {}
                    ).get("validation")
                    if validation_diagnostics:
                        teacher_time_diagnostics[other] = validation_diagnostics

        if args.skip_viz:
            generate_earthquake_density_plots(
                observed_train_points=train_initial.detach().cpu(),
                observed_validation_points=validation_initial.detach().cpu(),
                generated_by_teacher=teacher_generated,
                output_dir=viz_dir,
            )
        else:
            generate_earthquake_smoke_plots(
                observed_points=torch.cat(
                    (train_initial, validation_initial),
                    dim=0,
                ).detach().cpu(),
                observed_train_points=train_initial.detach().cpu(),
                observed_test_points=validation_initial.detach().cpu(),
                generated_by_teacher=teacher_generated,
                training_history_by_teacher=teacher_history,
                time_diagnostics_by_teacher=teacher_time_diagnostics,
                output_dir=viz_dir,
            )


if __name__ == "__main__":
    main()
