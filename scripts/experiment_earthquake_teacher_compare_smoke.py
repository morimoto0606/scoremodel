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
    s2_batched_discrete_malliavin_teacher,
    s2_discrete_malliavin_teacher,
    s2_grw_endpoint,
    s2_heat_kernel_score,
    s2_reverse_grw,
    s2_varadhan_score,
)
from scoremodel_ext.manifold.beta_schedule import LinearBetaSchedule
from scoremodel_ext.manifold.upstream_style_score import (
    UpstreamStyleScoreModel,
    build_upstream_style_score_model,
    train_s2_upstream_style_score_model,
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
    parser.add_argument(
        "--score-parameterization",
        choices=("effective_score", "upstream_scaled_score"),
        default="effective_score",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("upstream/riemannian-score-sde/data/quakes_all.csv"),
    )
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--validation-size", type=int, default=128)
    parser.add_argument("--n-steps", type=int, default=8)
    parser.add_argument(
        "--teacher-implementation",
        choices=("scalar", "batched"),
        default="scalar",
    )
    parser.add_argument(
        "--teacher-batch-size",
        type=int,
        choices=(1, 4, 8, 16),
        default=4,
    )
    parser.add_argument("--minimum-time", type=float, default=0.05)
    parser.add_argument("--maximum-time", type=float, default=0.3)
    parser.add_argument(
        "--beta-schedule",
        choices=("legacy-unit", "linear"),
        default="legacy-unit",
    )
    parser.add_argument("--beta-0", type=float, default=0.001)
    parser.add_argument("--beta-f", type=float, default=5.0)
    parser.add_argument("--beta-t0", type=float, default=0.0)
    parser.add_argument("--beta-tf", type=float, default=1.0)
    parser.add_argument(
        "--time-sampling",
        choices=("uniform", "curated-lowt"),
        default="uniform",
    )
    parser.add_argument("--time-samples-path", type=Path, default=None)
    parser.add_argument("--validation-time-samples-path", type=Path, default=None)
    parser.add_argument("--teacher-initial-points-path", type=Path, default=None)
    parser.add_argument("--teacher-noises-path", type=Path, default=None)
    parser.add_argument("--teacher-start-index", type=int, default=None)
    parser.add_argument("--teacher-end-index", type=int, default=None)
    parser.add_argument("--teacher-dataset-only", action="store_true")
    parser.add_argument("--prepare-teacher-inputs-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--training-unit",
        choices=("epochs", "updates"),
        default="epochs",
    )
    parser.add_argument("--updates", type=int, default=0)
    parser.add_argument("--warmup-updates", type=int, default=0)
    parser.add_argument(
        "--lr-scheduler",
        choices=("constant", "cosine"),
        default="constant",
    )
    parser.add_argument("--ema-rate", type=float, default=0.0)
    parser.add_argument("--use-ema-for-validation", action="store_true")
    parser.add_argument("--use-ema-for-reverse", action="store_true")
    parser.add_argument("--checkpoint-every-updates", type=int, default=0)
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
    parser.add_argument("--replay-original-reverse-artifacts", action="store_true")
    parser.add_argument("--covariance-regularization", type=float, default=1e-6)
    parser.add_argument("--heat-terms", type=int, default=80)
    parser.add_argument("--viz-output-dir", type=Path, default=None)
    parser.add_argument("--skip-viz", action="store_true")
    args = parser.parse_args()
    if args.skip_training and args.model_path is None:
        parser.error("--skip-training requires --model-path")
    if args.skip_teacher_generation and not args.skip_training:
        parser.error("--skip-teacher-generation requires --skip-training")
    if args.replay_original_reverse_artifacts and not args.skip_training:
        parser.error("--replay-original-reverse-artifacts requires --skip-training")
    if not args.skip_training and args.teacher is None:
        parser.error("--teacher is required unless --skip-training is used")
    if args.teacher_implementation == "batched" and args.teacher != "malliavin":
        parser.error("--teacher-implementation batched requires --teacher malliavin")
    if (
        args.score_parameterization == "upstream_scaled_score"
        and args.teacher not in {"heat", "malliavin"}
    ):
        parser.error(
            "--score-parameterization upstream_scaled_score requires "
            "--teacher heat or malliavin"
        )
    shard_bounds_given = (
        args.teacher_start_index is not None or args.teacher_end_index is not None
    )
    if shard_bounds_given and not args.teacher_dataset_only:
        parser.error(
            "--teacher-start-index/--teacher-end-index require "
            "--teacher-dataset-only"
        )
    if args.teacher_dataset_only:
        if args.teacher != "malliavin":
            parser.error("--teacher-dataset-only currently requires --teacher malliavin")
        if args.teacher_start_index is None or args.teacher_end_index is None:
            parser.error(
                "--teacher-dataset-only requires --teacher-start-index and "
                "--teacher-end-index"
            )
        required_paths = {
            "--teacher-initial-points-path": args.teacher_initial_points_path,
            "--time-samples-path": args.time_samples_path,
            "--validation-time-samples-path": args.validation_time_samples_path,
            "--teacher-noises-path": args.teacher_noises_path,
        }
        missing_paths = [name for name, value in required_paths.items() if value is None]
        if missing_paths:
            parser.error(
                "--teacher-dataset-only requires saved inputs: "
                + ", ".join(missing_paths)
            )
        if args.skip_training or args.skip_teacher_generation:
            parser.error(
                "--teacher-dataset-only cannot be combined with skip-training flags"
            )
    if args.prepare_teacher_inputs_only and (
        args.teacher_dataset_only or args.skip_training or args.skip_teacher_generation
    ):
        parser.error(
            "--prepare-teacher-inputs-only cannot be combined with dataset-only "
            "or skip-training flags"
        )
    if args.training_unit == "updates" and args.updates < 1:
        parser.error("--training-unit updates requires --updates > 0")
    if args.warmup_updates < 0:
        parser.error("--warmup-updates must be non-negative")
    if args.checkpoint_every_updates < 0:
        parser.error("--checkpoint-every-updates must be non-negative")
    if not 0.0 <= args.ema_rate < 1.0:
        parser.error("--ema-rate must be in [0, 1)")
    if (args.use_ema_for_validation or args.use_ema_for_reverse) and args.ema_rate <= 0:
        parser.error("EMA model selection requires --ema-rate > 0")
    return args


def to_dtype(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def resolve_device(name: str) -> str:
    if name == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return name


def build_beta_schedule(
    name: str,
    *,
    beta_0: float,
    beta_f: float,
    beta_t0: float,
    beta_tf: float,
) -> LinearBetaSchedule | None:
    """Build the optional schedule; ``None`` is the exact legacy code path."""

    if name == "legacy-unit":
        return None
    if name != "linear":
        raise ValueError(f"unknown beta schedule: {name!r}")
    return LinearBetaSchedule(
        beta_0=beta_0,
        beta_f=beta_f,
        t0=beta_t0,
        tf=beta_tf,
    )


def beta_schedule_from_args(args: argparse.Namespace) -> LinearBetaSchedule | None:
    return build_beta_schedule(
        getattr(args, "beta_schedule", "legacy-unit"),
        beta_0=getattr(args, "beta_0", 0.001),
        beta_f=getattr(args, "beta_f", 5.0),
        beta_t0=getattr(args, "beta_t0", 0.0),
        beta_tf=getattr(args, "beta_tf", 1.0),
    )


def beta_schedule_from_run_config(
    run_config: Dict[str, object],
) -> LinearBetaSchedule | None:
    """Restore a schedule, treating pre-schedule runs as exact legacy runs."""

    return build_beta_schedule(
        str(run_config.get("beta_schedule", "legacy-unit")),
        beta_0=float(run_config.get("beta_0", 0.001)),
        beta_f=float(run_config.get("beta_f", 5.0)),
        beta_t0=float(run_config.get("beta_t0", 0.0)),
        beta_tf=float(run_config.get("beta_tf", 1.0)),
    )


def validate_beta_schedule_time_range(
    beta_schedule: LinearBetaSchedule | None,
    *,
    minimum_time: float,
    maximum_time: float,
) -> None:
    if beta_schedule is None:
        return
    if minimum_time < beta_schedule.t0 or maximum_time > beta_schedule.tf:
        raise ValueError(
            "physical time range must lie within the linear beta schedule "
            f"[{beta_schedule.t0}, {beta_schedule.tf}]"
        )
    if beta_schedule.rescale_t(minimum_time) <= 0.0:
        raise ValueError("minimum physical time must have positive Brownian time")


def beta_schedule_metadata(
    beta_schedule: LinearBetaSchedule | None,
) -> Dict[str, object]:
    if beta_schedule is None:
        return {
            "beta_schedule": "legacy-unit",
            "beta_0": 0.001,
            "beta_f": 5.0,
            "beta_t0": 0.0,
            "beta_tf": 1.0,
        }
    return {
        "beta_schedule": "linear",
        "beta_0": beta_schedule.beta_0,
        "beta_f": beta_schedule.beta_f,
        "beta_t0": beta_schedule.t0,
        "beta_tf": beta_schedule.tf,
    }


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

    training_path = (
        "upstream_scaled_score"
        if run_config.get("score_parameterization") == "upstream_scaled_score"
        else "direct_score"
    )
    model = _build_saved_model_architecture(
        teacher=str(run_config["teacher"]),
        training_path=training_path,
        hidden=int(run_config["hidden"]),
        n_blocks=int(run_config["n_blocks"]),
        num_frequencies=int(run_config["num_frequencies"]),
        dtype=to_dtype(str(run_config["dtype"])),
        device=device,
        beta_0=float(run_config.get("beta_0", 0.001)),
        beta_f=float(run_config.get("beta_f", 5.0)),
        beta_t0=float(run_config.get("beta_t0", 0.0)),
        beta_tf=float(run_config.get("beta_tf", 1.0)),
    )
    return _load_saved_model_state(model, model_path)


def _build_saved_model_architecture(
    *,
    teacher: str,
    training_path: str = "direct_score",
    hidden: int,
    n_blocks: int,
    num_frequencies: int,
    dtype: torch.dtype,
    device: str,
    beta_0: float = 0.001,
    beta_f: float = 5.0,
    beta_t0: float = 0.0,
    beta_tf: float = 1.0,
):
    """Build the classes returned by the original two training functions."""

    if training_path == "upstream_scaled_score":
        zeros_x = torch.zeros(1, 3, dtype=dtype, device=device)
        ones_x = torch.ones(1, 3, dtype=dtype, device=device)
        zeros_t = torch.zeros(1, 1, dtype=dtype, device=device)
        ones_t = torch.ones(1, 1, dtype=dtype, device=device)
        return build_upstream_style_score_model(
            x_mean=zeros_x,
            x_std=ones_x,
            t_mean=zeros_t,
            t_std=ones_t,
            hidden=hidden,
            n_blocks=n_blocks,
            num_frequencies=num_frequencies,
            beta_schedule=LinearBetaSchedule(
                beta_0=beta_0,
                beta_f=beta_f,
                t0=beta_t0,
                tf=beta_tf,
            ),
            device=device,
            dtype=dtype,
        )
    normalized_model = _build_normalized_saved_model(
        hidden=hidden,
        n_blocks=n_blocks,
        num_frequencies=num_frequencies,
        dtype=dtype,
        device=device,
    )
    if teacher == "malliavin":
        return S2SkorokhodScoreModel(normalized_model).to(
            device=device,
            dtype=dtype,
        )
    return normalized_model


def _build_normalized_saved_model(
    *,
    hidden: int,
    n_blocks: int,
    num_frequencies: int,
    dtype: torch.dtype,
    device: str,
):
    """Construct the normalized network before the optional S2 wrapper."""

    network = MirafzaliSkorokhodNet(
        x_dim=3,
        out_dim=3,
        hidden=hidden,
        n_blocks=n_blocks,
        num_frequencies=num_frequencies,
    ).to(device=device, dtype=dtype)
    # These must be distinct tensors.  register_buffer keeps the supplied
    # storage, so reusing one zero/one tensor for both x and y causes the later
    # y_mean/y_std load_state_dict copies to overwrite x_mean/x_std as well.
    x_mean = torch.zeros(1, 3, dtype=dtype, device=device)
    x_std = torch.ones(1, 3, dtype=dtype, device=device)
    t_mean = torch.zeros(1, 1, dtype=dtype, device=device)
    t_std = torch.ones(1, 1, dtype=dtype, device=device)
    y_mean = torch.zeros(1, 3, dtype=dtype, device=device)
    y_std = torch.ones(1, 3, dtype=dtype, device=device)
    normalized_model = NormalizedSkorokhodModel(
        network,
        x_mean,
        x_std,
        t_mean,
        t_std,
        y_mean,
        y_std,
    ).to(device=device, dtype=dtype)
    return normalized_model


def _load_saved_model_state(model, model_path: Path):
    checkpoint = torch.load(model_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"saved model has no state_dict: {model_path}")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


def build_model_from_training_checkpoint(model_path: Path, *, device: str):
    """Rebuild using checkpoint metadata and the original training-path wrapper."""

    checkpoint = torch.load(model_path, map_location="cpu")
    required = (
        "teacher",
        "training_path",
        "state_dict",
        "hidden",
        "n_blocks",
        "num_frequencies",
        "dtype",
    )
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f"model checkpoint is missing fields: {missing}")
    training_path = str(checkpoint["training_path"])
    teacher = str(checkpoint["teacher"])
    expected_paths = (
        {"marginal_skorokhod", "upstream_scaled_score"}
        if teacher == "malliavin"
        else {"direct_score", "upstream_scaled_score"}
    )
    if training_path not in expected_paths:
        raise ValueError(
            f"checkpoint training_path={training_path!r}, expected one of {sorted(expected_paths)!r}"
        )
    model = _build_saved_model_architecture(
        teacher=teacher,
        training_path=training_path,
        hidden=int(checkpoint["hidden"]),
        n_blocks=int(checkpoint["n_blocks"]),
        num_frequencies=int(checkpoint["num_frequencies"]),
        dtype=to_dtype(str(checkpoint["dtype"])),
        device=device,
        beta_0=float(checkpoint.get("beta_0", 0.001)),
        beta_f=float(checkpoint.get("beta_f", 5.0)),
        beta_t0=float(checkpoint.get("beta_t0", 0.0)),
        beta_tf=float(checkpoint.get("beta_tf", 1.0)),
    )
    return _load_saved_model_state(model, model_path)


NORMALIZATION_BUFFER_NAMES = (
    "x_mean",
    "x_std",
    "t_mean",
    "t_std",
    "y_mean",
    "y_std",
    "beta_0",
    "beta_f",
    "beta_t0",
    "beta_tf",
)


def _append_normalization_stage(
    trace: Dict[str, object],
    *,
    stage: str,
    normalized_model: object,
    checkpoint_state: Dict[str, torch.Tensor],
    checkpoint_prefix: str,
) -> None:
    buffers = {}
    stage_matches = True
    for name in NORMALIZATION_BUFFER_NAMES:
        checkpoint_key = f"{checkpoint_prefix}{name}"
        if checkpoint_key not in checkpoint_state or not hasattr(normalized_model, name):
            continue
        checkpoint_value = checkpoint_state[checkpoint_key].detach().cpu()
        current_value = getattr(normalized_model, name).detach().cpu()
        same_shape = tuple(checkpoint_value.shape) == tuple(current_value.shape)
        same_dtype = checkpoint_value.dtype == current_value.dtype
        exact_equal = bool(
            same_shape and same_dtype and torch.equal(checkpoint_value, current_value)
        )
        if same_shape:
            max_abs_error = float(
                torch.max(
                    torch.abs(
                        checkpoint_value.to(torch.float64)
                        - current_value.to(torch.float64)
                    )
                )
            )
        else:
            max_abs_error = None
        stage_matches = stage_matches and exact_equal
        buffers[name] = {
            "checkpoint_key": checkpoint_key,
            "checkpoint_value": checkpoint_value.tolist(),
            "current_value": current_value.tolist(),
            "checkpoint_shape": list(checkpoint_value.shape),
            "current_shape": list(current_value.shape),
            "checkpoint_dtype": str(checkpoint_value.dtype),
            "current_dtype": str(current_value.dtype),
            "max_abs_error": max_abs_error,
            "exact_equal": exact_equal,
        }
    trace["stages"].append(
        {
            "stage": stage,
            "all_normalization_buffers_exact": stage_matches,
            "buffers": buffers,
        }
    )


def build_model_from_training_checkpoint_with_normalization_trace(
    model_path: Path,
    *,
    device: str,
) -> tuple[object, Dict[str, object], object, str]:
    """Rebuild the training model while tracing normalization-buffer stages."""

    checkpoint = torch.load(model_path, map_location="cpu")
    teacher = str(checkpoint["teacher"])
    dtype = to_dtype(str(checkpoint["dtype"]))
    if checkpoint.get("training_path") == "upstream_scaled_score":
        model = build_model_from_training_checkpoint(model_path, device=device)
        trace: Dict[str, object] = {
            "teacher": teacher,
            "training_path": "upstream_scaled_score",
            "model_path": str(model_path),
            "checkpoint_prefix": "",
            "stages": [],
        }
        _append_normalization_stage(
            trace,
            stage="1_constructor_and_load_immediately_after",
            normalized_model=model,
            checkpoint_state=checkpoint["state_dict"],
            checkpoint_prefix="",
        )
        return model, trace, model, ""
    normalized_model = _build_normalized_saved_model(
        hidden=int(checkpoint["hidden"]),
        n_blocks=int(checkpoint["n_blocks"]),
        num_frequencies=int(checkpoint["num_frequencies"]),
        dtype=dtype,
        device=device,
    )
    checkpoint_prefix = "skorokhod_network." if teacher == "malliavin" else ""
    trace: Dict[str, object] = {
        "teacher": teacher,
        "model_path": str(model_path),
        "checkpoint_prefix": checkpoint_prefix,
        "stages": [],
    }
    _append_normalization_stage(
        trace,
        stage="1_constructor_immediately_after",
        normalized_model=normalized_model,
        checkpoint_state=checkpoint["state_dict"],
        checkpoint_prefix=checkpoint_prefix,
    )

    if teacher == "malliavin":
        inner_state = {
            key.removeprefix(checkpoint_prefix): value
            for key, value in checkpoint["state_dict"].items()
            if key.startswith(checkpoint_prefix)
        }
    else:
        inner_state = checkpoint["state_dict"]
    normalized_model.load_state_dict(inner_state, strict=True)
    normalized_model.eval()
    _append_normalization_stage(
        trace,
        stage="2_load_state_dict_immediately_after",
        normalized_model=normalized_model,
        checkpoint_state=checkpoint["state_dict"],
        checkpoint_prefix=checkpoint_prefix,
    )

    if teacher == "malliavin":
        model = S2SkorokhodScoreModel(normalized_model).to(
            device=device,
            dtype=dtype,
        )
    else:
        model = normalized_model
    model.eval()
    _append_normalization_stage(
        trace,
        stage="3_wrapper_immediately_after",
        normalized_model=normalized_model,
        checkpoint_state=checkpoint["state_dict"],
        checkpoint_prefix=checkpoint_prefix,
    )
    return model, trace, normalized_model, checkpoint_prefix


def build_model_from_checkpoint_metadata(model_path: Path, *, device: str):
    """Rebuild solely from architecture metadata stored inside model.pt."""

    checkpoint = torch.load(model_path, map_location="cpu")
    model = _build_saved_model_architecture(
        teacher=str(checkpoint["teacher"]),
        training_path=str(checkpoint.get("training_path", "direct_score")),
        hidden=int(checkpoint["hidden"]),
        n_blocks=int(checkpoint["n_blocks"]),
        num_frequencies=int(checkpoint["num_frequencies"]),
        dtype=to_dtype(str(checkpoint["dtype"])),
        device=device,
        beta_0=float(checkpoint.get("beta_0", 0.001)),
        beta_f=float(checkpoint.get("beta_f", 5.0)),
        beta_t0=float(checkpoint.get("beta_t0", 0.0)),
        beta_tf=float(checkpoint.get("beta_tf", 1.0)),
    )
    return _load_saved_model_state(model, model_path)


def checkpoint_inventory(model_path: Path) -> Dict[str, object]:
    """Describe checkpoint metadata and every state-dict tensor shape."""

    checkpoint = torch.load(model_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"saved model has no state_dict: {model_path}")
    metadata = {
        key: {
            "type": type(value).__name__,
            **({"shape": list(value.shape)} if isinstance(value, torch.Tensor) else {}),
            **(
                {"value": value}
                if isinstance(value, (str, int, float, bool)) or value is None
                else {}
            ),
        }
        for key, value in checkpoint.items()
        if key != "state_dict"
    }
    state_dict = checkpoint["state_dict"]
    return {
        "checkpoint_keys": list(checkpoint.keys()),
        "metadata": metadata,
        "state_dict": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for key, value in state_dict.items()
        },
    }


def compare_checkpoint_state(model, model_path: Path) -> Dict[str, object]:
    """Compare every restored state tensor with the serialized checkpoint."""

    checkpoint_state = torch.load(model_path, map_location="cpu")["state_dict"]
    restored_state = model.state_dict()
    missing_keys = [key for key in checkpoint_state if key not in restored_state]
    unexpected_keys = [key for key in restored_state if key not in checkpoint_state]
    rows = []
    overall_max_abs_error = 0.0
    first_mismatching_key = None
    for key, checkpoint_value in checkpoint_state.items():
        restored_value = restored_state.get(key)
        if restored_value is None:
            continue
        same_shape = tuple(checkpoint_value.shape) == tuple(restored_value.shape)
        if same_shape:
            checkpoint_cpu = checkpoint_value.detach().cpu()
            restored_cpu = restored_value.detach().cpu()
            max_abs_error = float(
                torch.max(
                    torch.abs(
                        checkpoint_cpu.to(torch.float64)
                        - restored_cpu.to(torch.float64)
                    )
                )
            )
            exact_equal = bool(
                checkpoint_cpu.dtype == restored_cpu.dtype
                and torch.equal(checkpoint_cpu, restored_cpu)
            )
            overall_max_abs_error = max(overall_max_abs_error, max_abs_error)
        else:
            max_abs_error = None
            exact_equal = False
        if not exact_equal and first_mismatching_key is None:
            first_mismatching_key = key
        rows.append(
            {
                "key": key,
                "checkpoint_shape": list(checkpoint_value.shape),
                "restored_shape": list(restored_value.shape),
                "checkpoint_dtype": str(checkpoint_value.dtype),
                "restored_dtype": str(restored_value.dtype),
                "max_abs_error": max_abs_error,
                "exact_equal": exact_equal,
            }
        )
    if first_mismatching_key is None:
        first_mismatching_key = (missing_keys + unexpected_keys or [None])[0]
    return {
        "keys": rows,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "overall_max_abs_error": overall_max_abs_error,
        "first_mismatching_key": first_mismatching_key,
    }


def finalize_normalization_trace(trace: Dict[str, object]) -> None:
    """Record the first post-load stage where normalization stops matching."""

    stages = trace["stages"]
    trace["first_stage_not_matching_checkpoint"] = next(
        (
            stage["stage"]
            for stage in stages
            if not stage["all_normalization_buffers_exact"]
        ),
        None,
    )
    trace["first_post_load_mismatch_stage"] = next(
        (
            stage["stage"]
            for stage in stages[1:]
            if not stage["all_normalization_buffers_exact"]
        ),
        None,
    )


def require_exact_checkpoint_state(model, model_path: Path) -> Dict[str, object]:
    """Fail unless every final model state tensor exactly matches model.pt."""

    comparison = compare_checkpoint_state(model, model_path)
    if (
        comparison["missing_keys"]
        or comparison["unexpected_keys"]
        or comparison["first_mismatching_key"] is not None
        or comparison["overall_max_abs_error"] != 0.0
    ):
        raise AssertionError(
            "final replay model does not exactly match checkpoint state: "
            f"first_mismatching_key={comparison['first_mismatching_key']!r}, "
            f"overall_max_abs_error={comparison['overall_max_abs_error']}"
        )
    return comparison


def compare_model_reconstruction_paths(
    *,
    teacher: str,
    run_config: Dict[str, object],
    checkpoint: Dict[str, object],
    models: Dict[str, object],
) -> Dict[str, object]:
    """Evaluate all saved-model reconstruction paths on fixed inputs."""

    dtype = to_dtype(str(checkpoint["dtype"]))
    device = next(models["A_run_config"].parameters()).device
    times = torch.tensor([0.005, 0.05, 0.1, 0.3], dtype=dtype, device=device)
    points = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=dtype,
        device=device,
    )
    points = normalize(points)
    with torch.no_grad():
        outputs = {
            name: model(times, points).detach().cpu()
            for name, model in models.items()
        }
    names = list(outputs)
    pairwise = {}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            pairwise[f"{left}_vs_{right}"] = float(
                torch.max(torch.abs(outputs[left] - outputs[right]))
            )
    metadata_fields = ("teacher", "hidden", "n_blocks", "num_frequencies", "dtype")
    metadata_mismatches = {
        key: {
            "run_config": run_config.get(key),
            "checkpoint": checkpoint.get(key),
        }
        for key in metadata_fields
        if run_config.get(key) != checkpoint.get(key)
    }
    return {
        "teacher": teacher,
        "input_t": times.detach().cpu().tolist(),
        "input_x": points.detach().cpu().tolist(),
        "outputs": {name: value.tolist() for name, value in outputs.items()},
        "pairwise_max_abs_error": pairwise,
        "metadata_mismatches": metadata_mismatches,
    }


def checkpoint_state_max_abs_error(model, model_path: Path) -> float:
    """Return the largest tensor error between a model and its checkpoint."""

    checkpoint = torch.load(model_path, map_location="cpu")["state_dict"]
    restored = model.state_dict()
    if checkpoint.keys() != restored.keys():
        raise ValueError("restored model state_dict keys differ from checkpoint")
    return max(
        float(
            torch.max(
                torch.abs(
                    restored[key].detach().cpu().to(torch.float64)
                    - checkpoint[key].detach().cpu().to(torch.float64)
                )
            )
        )
        for key in checkpoint
    )


def load_original_reverse_artifact(
    path: Path,
    *,
    reverse_steps: int,
    n_generated_samples: int,
    dtype: torch.dtype,
    device: str,
    output_path: Path,
) -> torch.Tensor:
    """Load an original run's reverse noise without pooling or aggregation."""

    if not path.is_file():
        raise FileNotFoundError(f"missing original reverse noise: {path}")
    payload = torch.load(path, map_location="cpu")
    noise = payload["reverse_noise"] if isinstance(payload, dict) else payload
    expected_shape = (reverse_steps, n_generated_samples, 3)
    if tuple(noise.shape) != expected_shape:
        raise ValueError(
            f"original reverse noise must have shape {expected_shape}, "
            f"got {tuple(noise.shape)} from {path}"
        )
    noise = noise.to(device=device, dtype=dtype)
    torch.save(noise.detach().cpu(), output_path)
    return noise


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


MALLIAVIN_TEACHER_DETAIL_KEYS = (
    "endpoint",
    "endpoint_jacobian",
    "covariance",
    "covering",
    "divergence_term",
    "skorokhod",
    "score_weight",
)


def _is_out_of_memory_error(error: RuntimeError) -> bool:
    oom_type = getattr(torch, "OutOfMemoryError", None)
    return (
        (oom_type is not None and isinstance(error, oom_type))
        or "out of memory" in str(error).lower()
    )


def compute_batched_malliavin_teacher_chunks(
    *,
    initial_points: torch.Tensor,
    times: torch.Tensor,
    noises: torch.Tensor,
    batch_size: int,
    covariance_regularization: float,
) -> tuple[list[object], list[int]]:
    """Evaluate ordered chunks, halving only the failed chunk after CUDA OOM."""

    if batch_size < 1:
        raise ValueError("teacher batch_size must be positive")
    if times.shape != (initial_points.shape[0],):
        raise ValueError("times must have shape [n_samples]")
    if noises.shape[0] != initial_points.shape[0]:
        raise ValueError("noises sample dimension does not match")

    chunks = []
    effective_batch_sizes = []
    cursor = 0
    active_batch_size = batch_size
    while cursor < initial_points.shape[0]:
        current_size = min(active_batch_size, initial_points.shape[0] - cursor)
        while True:
            end = cursor + current_size
            try:
                teacher = s2_batched_discrete_malliavin_teacher(
                    initial_points[cursor:end],
                    noises[cursor:end],
                    times[cursor:end],
                    covariance_regularization=covariance_regularization,
                )
                break
            except RuntimeError as error:
                if not _is_out_of_memory_error(error) or current_size == 1:
                    raise
                failed_size = current_size
                current_size = max(1, current_size // 2)
                active_batch_size = min(active_batch_size, current_size)
                if initial_points.device.type == "cuda":
                    torch.cuda.empty_cache()
                warnings.warn(
                    "Malliavin teacher batch ran out of memory; retrying the "
                    f"same samples with batch size {current_size} "
                    f"instead of {failed_size}",
                    RuntimeWarning,
                    stacklevel=1,
                )
        chunks.append(teacher)
        effective_batch_sizes.append(current_size)
        cursor += current_size
    return chunks, effective_batch_sizes


def build_malliavin_teacher_dataset_batched(
    *,
    initial_points: torch.Tensor,
    times: torch.Tensor,
    noises: torch.Tensor,
    batch_size: int,
    covariance_regularization: float,
    beta_schedule: LinearBetaSchedule | None = None,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], list[int]]:
    """Build the ordinary dataset with exact sample-batched teacher chunks."""

    brownian_times = (
        times if beta_schedule is None else beta_schedule.rescale_t(times)
    )
    chunks, effective_batch_sizes = compute_batched_malliavin_teacher_chunks(
        initial_points=initial_points,
        times=brownian_times,
        noises=noises,
        batch_size=batch_size,
        covariance_regularization=covariance_regularization,
    )

    def concatenate(attribute: str) -> torch.Tensor:
        return torch.cat(
            [getattr(chunk, attribute).detach() for chunk in chunks],
            dim=0,
        )

    details = {
        key: concatenate(key) for key in MALLIAVIN_TEACHER_DETAIL_KEYS
    }
    dataset = {
        "initial_point": initial_points.detach(),
        "time": times.detach(),
        "noise": noises.detach(),
        "endpoint": details["endpoint"],
        "score_target": details["score_weight"],
        "skorokhod": details["skorokhod"],
    }
    return dataset, details, effective_batch_sizes


def _load_split_tensor_payload(
    path: Path,
    *,
    train_key: str,
    validation_key: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"missing saved teacher input: {path}")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"teacher input must be a dictionary: {path}")
    missing = [key for key in (train_key, validation_key) if key not in payload]
    if missing:
        raise ValueError(f"teacher input {path} is missing keys: {missing}")
    train = payload[train_key]
    validation = payload[validation_key]
    if not isinstance(train, torch.Tensor) or not isinstance(validation, torch.Tensor):
        raise TypeError(f"teacher input values must be tensors: {path}")
    return train, validation


def load_saved_teacher_shard_inputs(
    *,
    initial_points_path: Path,
    train_times_path: Path,
    validation_times_path: Path,
    noises_path: Path,
    train_size: int,
    validation_size: int,
    n_steps: int,
    dtype: torch.dtype,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load fixed shard inputs without drawing any random variables."""

    train_initial, validation_initial = _load_split_tensor_payload(
        initial_points_path,
        train_key="train_initial_points",
        validation_key="validation_initial_points",
    )
    if not train_times_path.is_file() or not validation_times_path.is_file():
        raise FileNotFoundError("missing saved train/validation teacher times")
    train_times = _time_tensor_from_payload(
        torch.load(train_times_path, map_location="cpu"),
        key="train_times",
    )
    validation_times = _time_tensor_from_payload(
        torch.load(validation_times_path, map_location="cpu"),
        key="validation_times",
    )
    train_noises, validation_noises = _load_split_tensor_payload(
        noises_path,
        train_key="train_noises",
        validation_key="validation_noises",
    )

    expected = {
        "train_initial_points": (train_initial, (train_size, 3)),
        "validation_initial_points": (validation_initial, (validation_size, 3)),
        "train_times": (train_times, (train_size,)),
        "validation_times": (validation_times, (validation_size,)),
        "train_noises": (train_noises, (train_size, n_steps, 3)),
        "validation_noises": (
            validation_noises,
            (validation_size, n_steps, 3),
        ),
    }
    for name, (tensor, shape) in expected.items():
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
        if tensor.dtype != dtype:
            raise ValueError(
                f"{name} must have dtype {dtype}, got {tensor.dtype}; "
                "shard workers do not convert saved inputs"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} contains non-finite values")

    initial_points = torch.cat((train_initial, validation_initial), dim=0)
    times = torch.cat((train_times, validation_times), dim=0)
    noises = torch.cat((train_noises, validation_noises), dim=0)
    return (
        initial_points.to(device=device),
        times.to(device=device),
        noises.to(device=device),
    )


def build_malliavin_teacher_shard(
    *,
    initial_points: torch.Tensor,
    times: torch.Tensor,
    noises: torch.Tensor,
    start: int,
    end: int,
    train_size: int,
    validation_size: int,
    covariance_regularization: float,
    teacher_implementation: str = "scalar",
    teacher_batch_size: int = 4,
    beta_schedule: LinearBetaSchedule | None = None,
) -> Dict[str, object]:
    """Run one Malliavin implementation over a global shard index range."""

    total_size = train_size + validation_size
    if initial_points.shape != (total_size, 3):
        raise ValueError("combined initial_points shape does not match split sizes")
    if times.shape != (total_size,):
        raise ValueError("combined times shape does not match split sizes")
    if noises.ndim != 3 or noises.shape[0] != total_size or noises.shape[2] != 3:
        raise ValueError("combined noises must have shape [total_size, n_steps, 3]")
    if not 0 <= start < end <= total_size:
        raise ValueError(
            f"teacher shard range must satisfy 0 <= start < end <= {total_size}"
        )
    input_dtype = initial_points.dtype
    if times.dtype != input_dtype or noises.dtype != input_dtype:
        raise ValueError("initial_points, times, and noises must have one dtype")

    if teacher_implementation == "batched":
        dataset, teacher_details, effective_batch_sizes = (
            build_malliavin_teacher_dataset_batched(
                initial_points=initial_points[start:end],
                times=times[start:end],
                noises=noises[start:end],
                batch_size=teacher_batch_size,
                covariance_regularization=covariance_regularization,
                beta_schedule=beta_schedule,
            )
        )
        dataset = {key: value.detach().cpu() for key, value in dataset.items()}
        teacher_details = {
            key: value.detach().cpu() for key, value in teacher_details.items()
        }
    elif teacher_implementation == "scalar":
        dataset_lists: Dict[str, list[torch.Tensor]] = {
            "initial_point": [],
            "time": [],
            "noise": [],
            "endpoint": [],
            "score_target": [],
            "skorokhod": [],
        }
        detail_lists: Dict[str, list[torch.Tensor]] = {
            key: [] for key in MALLIAVIN_TEACHER_DETAIL_KEYS
        }
        for global_index in range(start, end):
            initial_point = initial_points[global_index]
            terminal_time = times[global_index]
            noise = noises[global_index]
            brownian_time = (
                terminal_time
                if beta_schedule is None
                else beta_schedule.rescale_t(terminal_time)
            )
            teacher_state = s2_discrete_malliavin_teacher(
                initial_point,
                noise,
                float(brownian_time.detach().cpu()),
                covariance_regularization=covariance_regularization,
                vectorize_jacobian=True,
            )
            dataset_lists["initial_point"].append(initial_point)
            dataset_lists["time"].append(terminal_time)
            dataset_lists["noise"].append(noise)
            dataset_lists["endpoint"].append(teacher_state.endpoint)
            dataset_lists["score_target"].append(teacher_state.score_weight)
            dataset_lists["skorokhod"].append(teacher_state.skorokhod)
            for key in MALLIAVIN_TEACHER_DETAIL_KEYS:
                detail_lists[key].append(getattr(teacher_state, key))
        dataset = {
            key: torch.stack(values).detach().cpu()
            for key, values in dataset_lists.items()
        }
        teacher_details = {
            key: torch.stack(values).detach().cpu()
            for key, values in detail_lists.items()
        }
        effective_batch_sizes = [1] * (end - start)
    else:
        raise ValueError(f"unknown teacher implementation: {teacher_implementation!r}")

    dtype_name = str(input_dtype).removeprefix("torch.")
    return {
        "format_version": 1,
        "teacher": "malliavin",
        "start": start,
        "end": end,
        "total_size": total_size,
        "train_size": train_size,
        "validation_size": validation_size,
        "dataset_keys": list(dataset),
        "detail_keys": list(teacher_details),
        "dtype": dtype_name,
        "teacher_implementation": teacher_implementation,
        "requested_teacher_batch_size": teacher_batch_size,
        "effective_teacher_batch_sizes": effective_batch_sizes,
        **beta_schedule_metadata(beta_schedule),
        "global_indices": torch.arange(start, end, dtype=torch.int64),
        "dataset": dataset,
        "teacher_details": teacher_details,
    }


def run_teacher_dataset_shard(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    log,
) -> None:
    """Generate and save exactly one Malliavin teacher shard, then stop."""

    dtype = to_dtype(args.dtype)
    device = resolve_device(args.device)
    beta_schedule = beta_schedule_from_args(args)
    validate_beta_schedule_time_range(
        beta_schedule,
        minimum_time=getattr(args, "minimum_time", 0.05),
        maximum_time=getattr(args, "maximum_time", 0.3),
    )
    initial_points, times, noises = load_saved_teacher_shard_inputs(
        initial_points_path=args.teacher_initial_points_path,
        train_times_path=args.time_samples_path,
        validation_times_path=args.validation_time_samples_path,
        noises_path=args.teacher_noises_path,
        train_size=args.train_size,
        validation_size=args.validation_size,
        n_steps=args.n_steps,
        dtype=dtype,
        device=device,
    )
    start = int(args.teacher_start_index)
    end = int(args.teacher_end_index)
    teacher_implementation = getattr(args, "teacher_implementation", "scalar")
    teacher_batch_size = getattr(args, "teacher_batch_size", 4)
    log(
        f"generating {teacher_implementation} Malliavin teacher shard "
        f"[{start}, {end}) on {device}"
    )
    started = time.perf_counter()
    payload = build_malliavin_teacher_shard(
        initial_points=initial_points,
        times=times,
        noises=noises,
        start=start,
        end=end,
        train_size=args.train_size,
        validation_size=args.validation_size,
        covariance_regularization=args.covariance_regularization,
        teacher_implementation=teacher_implementation,
        teacher_batch_size=teacher_batch_size,
        beta_schedule=beta_schedule,
    )
    payload["beta_schedule"] = getattr(args, "beta_schedule", "legacy-unit")
    payload["beta_0"] = getattr(args, "beta_0", 0.001)
    payload["beta_f"] = getattr(args, "beta_f", 5.0)
    payload["beta_t0"] = getattr(args, "beta_t0", 0.0)
    payload["beta_tf"] = getattr(args, "beta_tf", 1.0)
    payload["generation_seconds"] = time.perf_counter() - started
    shard_path = output_dir / f"teacher_dataset_shard_{start:06d}_{end:06d}.pt"
    torch.save(payload, shard_path)
    with (output_dir / "teacher_shard_config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "teacher": "malliavin",
                "start": start,
                "end": end,
                "total_size": args.train_size + args.validation_size,
                "train_size": args.train_size,
                "validation_size": args.validation_size,
                "n_steps": args.n_steps,
                "dtype": args.dtype,
                "device": device,
                "teacher_implementation": teacher_implementation,
                "requested_teacher_batch_size": teacher_batch_size,
                "beta_schedule": getattr(args, "beta_schedule", "legacy-unit"),
                "beta_0": getattr(args, "beta_0", 0.001),
                "beta_f": getattr(args, "beta_f", 5.0),
                "beta_t0": getattr(args, "beta_t0", 0.0),
                "beta_tf": getattr(args, "beta_tf", 1.0),
                "effective_teacher_batch_sizes": payload[
                    "effective_teacher_batch_sizes"
                ],
                "initial_points_path": str(args.teacher_initial_points_path),
                "train_times_path": str(args.time_samples_path),
                "validation_times_path": str(args.validation_time_samples_path),
                "teacher_noises_path": str(args.teacher_noises_path),
                "shard_path": str(shard_path),
                "generation_seconds": payload["generation_seconds"],
            },
            handle,
            indent=2,
        )
    log(f"saved teacher shard: {shard_path}")


def build_teacher_dataset(
    *,
    initial_points: torch.Tensor,
    times: torch.Tensor,
    noises: torch.Tensor,
    teacher: str,
    covariance_regularization: float,
    heat_terms: int,
    beta_schedule: LinearBetaSchedule | None = None,
) -> Dict[str, torch.Tensor]:
    endpoints = []
    score_target = []
    skorokhod = []

    for initial_point, terminal_time, noise in zip(initial_points, times, noises):
        terminal_time_float = float(terminal_time.detach().cpu())
        brownian_time = (
            terminal_time
            if beta_schedule is None
            else beta_schedule.rescale_t(terminal_time)
        )
        brownian_time_float = (
            terminal_time_float
            if beta_schedule is None
            else float(brownian_time.detach().cpu())
        )
        if teacher == "malliavin":
            teacher_state = s2_discrete_malliavin_teacher(
                initial_point,
                noise,
                brownian_time_float,
                covariance_regularization=covariance_regularization,
                vectorize_jacobian=True,
            )
            endpoints.append(teacher_state.endpoint)
            score_target.append(teacher_state.score_weight)
            skorokhod.append(teacher_state.skorokhod)
        elif teacher == "heat":
            endpoint = s2_grw_endpoint(initial_point, noise, brownian_time_float)
            endpoints.append(endpoint)
            score_target.append(
                s2_heat_kernel_score(
                    initial_point,
                    endpoint,
                    brownian_time_float,
                    n_terms=heat_terms,
                )
            )
        else:
            endpoint = s2_grw_endpoint(initial_point, noise, brownian_time_float)
            endpoints.append(endpoint)
            score_target.append(
                s2_varadhan_score(initial_point, endpoint, brownian_time_float)
            )

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
        if isinstance(model, UpstreamStyleScoreModel):
            value = model.score_loss(
                dataset["time"],
                dataset["endpoint"],
                dataset["score_target"],
            )
        elif teacher == "malliavin":
            prediction = model.skorokhod_network(dataset["time"], dataset["endpoint"])
            target = dataset["skorokhod"]
            value = torch.mean((prediction - target) ** 2)
        else:
            prediction = model(dataset["time"], dataset["endpoint"])
            target = dataset["score_target"]
            value = torch.mean((prediction - target) ** 2)
    return float(value)


def select_online_or_ema_model(
    online_model,
    ema_model,
    *,
    use_ema: bool,
    purpose: str,
):
    if not use_ema:
        return online_model, "online"
    if ema_model is None:
        raise ValueError(f"EMA was requested for {purpose}, but EMA is disabled")
    return ema_model, "ema"


def evaluate_selected_validation_loss(
    online_model,
    ema_model,
    dataset: Dict[str, torch.Tensor],
    *,
    teacher: str,
    use_ema: bool,
) -> tuple[float, float | None, float | None, str]:
    """Evaluate only the model selected for validation."""

    validation_model, model_source = select_online_or_ema_model(
        online_model,
        ema_model,
        use_ema=use_ema,
        purpose="validation",
    )
    validation_loss = evaluate_dataset_loss(
        validation_model,
        dataset,
        teacher=teacher,
    )
    validation_loss_online = (
        validation_loss if model_source == "online" else None
    )
    validation_loss_ema = validation_loss if model_source == "ema" else None
    return (
        validation_loss,
        validation_loss_online,
        validation_loss_ema,
        model_source,
    )


def build_model_checkpoint_payload(
    *,
    selected_model,
    online_model,
    ema_model,
    model_source: str,
    teacher: str,
    training_path: str,
    hidden: int,
    n_blocks: int,
    num_frequencies: int,
    dtype: str,
    training_state: Dict[str, object],
    training_unit: str,
    requested_epochs: int,
    requested_updates: int,
    base_learning_rate: float,
    warmup_updates: int,
    lr_scheduler: str,
    ema_rate: float,
    use_ema_for_validation: bool,
    use_ema_for_reverse: bool,
    checkpoint_every_updates: int,
    beta_schedule: str,
    beta_0: float,
    beta_f: float,
    beta_t0: float,
    beta_tf: float,
) -> Dict[str, object]:
    """Build a backward-compatible final model checkpoint."""

    return {
        "teacher": teacher,
        "training_path": training_path,
        "score_parameterization": (
            "upstream_scaled_score"
            if training_path == "upstream_scaled_score"
            else "effective_score"
        ),
        "state_dict": selected_model.state_dict(),
        "model_source": model_source,
        "online_state_dict": online_model.state_dict(),
        "ema_state_dict": (
            ema_model.state_dict() if ema_model is not None else None
        ),
        "optimizer_state_dict": training_state["optimizer_state_dict"],
        "scheduler_state": training_state["scheduler_state"],
        "current_update": training_state["current_update"],
        "current_epoch": training_state["current_epoch"],
        "training_unit": training_unit,
        "requested_total_updates": requested_updates,
        "requested_epochs": requested_epochs,
        "actual_optimizer_updates": training_state["actual_optimizer_updates"],
        "updates_per_epoch": training_state["updates_per_epoch"],
        "effective_epochs": training_state["effective_epochs"],
        "base_learning_rate": base_learning_rate,
        "warmup_updates": warmup_updates,
        "lr_scheduler": lr_scheduler,
        "initial_learning_rate": training_state["initial_learning_rate"],
        "peak_learning_rate": training_state["peak_learning_rate"],
        "final_learning_rate": training_state["final_learning_rate"],
        "learning_rate_trace": training_state["learning_rate_trace"],
        "ema_rate": ema_rate,
        "use_ema_for_validation": use_ema_for_validation,
        "use_ema_for_reverse": use_ema_for_reverse,
        "checkpoint_every_updates": checkpoint_every_updates,
        "normalization_state": training_state["normalization_state"],
        "hidden": hidden,
        "n_blocks": n_blocks,
        "num_frequencies": num_frequencies,
        "dtype": dtype,
        "beta_schedule": beta_schedule,
        "beta_0": beta_0,
        "beta_f": beta_f,
        "beta_t0": beta_t0,
        "beta_tf": beta_tf,
    }


def make_training_checkpoint_callback(
    *,
    output_dir: Path,
    args: argparse.Namespace,
):
    if args.checkpoint_every_updates <= 0:
        return None

    checkpoint_dir = output_dir / "training_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def complete_model_state(
        network_state: Dict[str, torch.Tensor] | None,
        normalization_state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor] | None:
        if network_state is None:
            return None
        prefix = "skorokhod_network." if args.teacher == "malliavin" else ""
        state = {
            f"{prefix}{key}": value
            for key, value in normalization_state.items()
        }
        state.update(
            {
                f"{prefix}net.{key}": value
                for key, value in network_state.items()
            }
        )
        return state

    def save_checkpoint(payload: Dict[str, object]) -> None:
        enriched = dict(payload)
        if "complete_online_model_state_dict" in enriched:
            online_state = enriched["complete_online_model_state_dict"]
            ema_state = enriched["complete_ema_model_state_dict"]
        else:
            online_state = complete_model_state(
                enriched["online_network_state_dict"],
                enriched["normalization_state"],
            )
            ema_state = complete_model_state(
                enriched["ema_network_state_dict"],
                enriched["normalization_state"],
            )
        model_source = (
            "ema" if args.use_ema_for_reverse and ema_state is not None else "online"
        )
        enriched.update(
            {
                "format_version": 1,
                "teacher": args.teacher,
                "training_path": (
                    "upstream_scaled_score"
                    if args.score_parameterization == "upstream_scaled_score"
                    else (
                        "marginal_skorokhod"
                        if args.teacher == "malliavin"
                        else "direct_score"
                    )
                ),
                "score_parameterization": args.score_parameterization,
                "training_unit": args.training_unit,
                "requested_epochs": args.epochs,
                "requested_updates": args.updates,
                "requested_total_updates": args.updates,
                "base_learning_rate": args.learning_rate,
                "checkpoint_every_updates": args.checkpoint_every_updates,
                "hidden": args.hidden,
                "n_blocks": args.n_blocks,
                "num_frequencies": args.num_frequencies,
                "dtype": args.dtype,
                "model_source": model_source,
                "state_dict": (
                    ema_state if model_source == "ema" else online_state
                ),
                "online_state_dict": online_state,
                "ema_state_dict": ema_state,
                "use_ema_for_validation": args.use_ema_for_validation,
                "use_ema_for_reverse": args.use_ema_for_reverse,
                "beta_schedule": args.beta_schedule,
                "beta_0": args.beta_0,
                "beta_f": args.beta_f,
                "beta_t0": args.beta_t0,
                "beta_tf": args.beta_tf,
            }
        )
        update = int(enriched["current_update"])
        torch.save(
            enriched,
            checkpoint_dir / f"checkpoint_update_{update:09d}.pt",
        )

    return save_checkpoint


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


def diagnostic_target_key_for_teacher(
    teacher: str,
    score_parameterization: str = "effective_score",
) -> str:
    if score_parameterization == "upstream_scaled_score":
        return "score_target"
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
    beta_schedule = beta_schedule_from_run_config(source_config)

    log(f"loading saved {teacher} model from {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu")
    model_used_for_reverse = str(checkpoint.get("model_source", "online"))
    if model_used_for_reverse not in {"online", "ema"}:
        raise ValueError(
            f"invalid model_source in checkpoint: {model_used_for_reverse!r}"
        )
    model_a = build_model_from_run_config(model_path, source_config, device=device)
    (
        model_b,
        normalization_trace,
        normalized_model_b,
        checkpoint_prefix_b,
    ) = build_model_from_training_checkpoint_with_normalization_trace(
        model_path,
        device=device,
    )
    model_c = build_model_from_checkpoint_metadata(model_path, device=device)
    _append_normalization_stage(
        normalization_trace,
        stage="4_fixed_input_evaluation_immediately_before",
        normalized_model=normalized_model_b,
        checkpoint_state=checkpoint["state_dict"],
        checkpoint_prefix=checkpoint_prefix_b,
    )
    model_output_comparison = compare_model_reconstruction_paths(
        teacher=teacher,
        run_config=source_config,
        checkpoint=checkpoint,
        models={
            "A_run_config": model_a,
            "B_training_path": model_b,
            "C_checkpoint_metadata": model_c,
        },
    )
    with (output_dir / "model_output_comparison.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(model_output_comparison, handle, indent=2)
    # Replay uses the original training-path wrapper and checkpoint metadata.
    model = model_b
    model_state_error = checkpoint_state_max_abs_error(model, model_path)
    inventory = checkpoint_inventory(model_path)
    with (output_dir / "checkpoint_inventory.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(inventory, handle, indent=2)
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
    if args.replay_original_reverse_artifacts and not terminal_path.is_file():
        raise FileNotFoundError(f"missing original terminal samples: {terminal_path}")
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

    if args.replay_original_reverse_artifacts:
        original_reverse_steps = int(source_config["reverse_steps"])
        if reverse_steps != original_reverse_steps:
            raise ValueError(
                "original-artifact replay requires reverse_steps to match "
                f"the source run ({original_reverse_steps})"
            )
        reverse_noise_path = (
            args.reverse_noise_path.expanduser().resolve()
            if args.reverse_noise_path is not None
            else source_model_dir / "reverse_noise.pt"
        )
        reverse_noise = load_original_reverse_artifact(
            reverse_noise_path,
            reverse_steps=reverse_steps,
            n_generated_samples=n_generated_samples,
            dtype=dtype,
            device=device,
            output_path=output_dir / "reverse_noise.pt",
        )
        reverse_noise_coupling = "original_run_artifact"
        reverse_noise_coupling_exact = True
    else:
        reverse_noise_path = (
            args.reverse_noise_path.expanduser().resolve()
            if args.reverse_noise_path is not None
            else source_model_dir / "reverse_noise_1000.pt"
        )
        reverse_noise = maybe_load_or_create_shared_reverse_noise(
            path=reverse_noise_path,
            output_path=output_dir / "reverse_noise.pt",
            reverse_steps=reverse_steps,
            n_generated_samples=n_generated_samples,
            dtype=dtype,
            device=device,
            seed=int(source_config["reverse_seed"]),
        )
        reverse_noise_coupling = (
            "linear_interpolation_of_cumulative_fine_brownian_path"
        )
        reverse_noise_coupling_exact = (
            MAX_REVERSE_NOISE_STEPS % reverse_steps == 0
        )

    _append_normalization_stage(
        normalization_trace,
        stage="5_reverse_sampling_immediately_before",
        normalized_model=normalized_model_b,
        checkpoint_state=checkpoint["state_dict"],
        checkpoint_prefix=checkpoint_prefix_b,
    )
    finalize_normalization_trace(normalization_trace)
    with (output_dir / "normalization_state_stages.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(normalization_trace, handle, indent=2)
    # This is the final gate immediately before reverse sampling.
    state_comparison = compare_checkpoint_state(model, model_path)
    with (output_dir / "checkpoint_state_comparison.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(state_comparison, handle, indent=2)
    require_exact_checkpoint_state(model, model_path)

    reverse_started = time.perf_counter()
    reverse_result = s2_reverse_grw(
        terminal_samples,
        build_score_fn(model),
        terminal_time=float(source_config["maximum_time"]),
        n_steps=reverse_steps,
        standard_noise=reverse_noise,
        minimum_forward_time=float(source_config["minimum_time"]),
        beta_schedule=beta_schedule,
        return_first_step=args.replay_original_reverse_artifacts,
        debug_output_dir=(
            output_dir if args.replay_original_reverse_artifacts else None
        ),
    )
    if args.replay_original_reverse_artifacts:
        generated, first_step_generated = reverse_result
        torch.save(
            first_step_generated.detach().cpu(),
            output_dir / "generated_samples_after_first_reverse_step.pt",
        )
    else:
        generated = reverse_result
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
    reproduction_max_abs_error = None
    reproduction_passed = False
    if args.replay_original_reverse_artifacts:
        reference_generated_path = source_model_dir / "generated_samples.pt"
        if not reference_generated_path.is_file():
            raise FileNotFoundError(
                "original-artifact replay requires the source generated samples: "
                f"{reference_generated_path}"
            )
        reference_generated = torch.load(
            reference_generated_path,
            map_location="cpu",
        ).to(dtype=dtype)
        if tuple(reference_generated.shape) != tuple(generated_cpu.shape):
            raise ValueError(
                "source generated samples have shape "
                f"{tuple(reference_generated.shape)}, expected {tuple(generated_cpu.shape)}"
            )
        reproduction_max_abs_error = float(
            torch.max(torch.abs(generated_cpu - reference_generated))
        )
        reproduction_passed = reproduction_max_abs_error <= 1e-12
        reproduction = {
            "teacher": teacher,
            "passed": reproduction_passed,
            "atol": 1e-12,
            "rtol": 0.0,
            "max_abs_error": reproduction_max_abs_error,
            "checkpoint_state_max_abs_error": model_state_error,
            "terminal_samples_path": str(terminal_path),
            "reverse_noise_path": str(reverse_noise_path),
            "reverse_steps": reverse_steps,
            "terminal_time": float(source_config["maximum_time"]),
            "minimum_forward_time": float(source_config["minimum_time"]),
            "dt": float(source_config["maximum_time"]) / reverse_steps,
            "first_forward_time": float(source_config["maximum_time"]),
            "last_forward_time": max(
                float(source_config["maximum_time"]) / reverse_steps,
                float(source_config["minimum_time"]),
            ),
        }
        with (output_dir / "original_run_reproduction.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(reproduction, handle, indent=2)
        if not reproduction_passed:
            raise AssertionError(
                "original 128-step generated samples were not reproduced: "
                f"max_abs_error={reproduction_max_abs_error:.6e}"
            )
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
        "original_reverse_artifact_replay": bool(
            args.replay_original_reverse_artifacts
        ),
        "original_reproduction_max_abs_error": reproduction_max_abs_error,
        "checkpoint_state_max_abs_error": model_state_error,
        "ablation_validated_against_original_run": reproduction_passed,
        "device": device,
        "dtype": dtype_name,
        "beta_schedule": str(source_config.get("beta_schedule", "legacy-unit")),
        "beta_0": float(source_config.get("beta_0", 0.001)),
        "beta_f": float(source_config.get("beta_f", 5.0)),
        "beta_t0": float(source_config.get("beta_t0", 0.0)),
        "beta_tf": float(source_config.get("beta_tf", 1.0)),
        "training_unit": str(source_config.get("training_unit", "epochs")),
        "requested_updates": int(source_config.get("requested_updates", 0)),
        "actual_optimizer_updates": int(
            source_config.get("actual_optimizer_updates", 0)
        ),
        "effective_epochs": float(source_config.get("effective_epochs", 0.0)),
        "warmup_updates": int(source_config.get("warmup_updates", 0)),
        "lr_scheduler": str(source_config.get("lr_scheduler", "constant")),
        "ema_enabled": checkpoint.get("ema_state_dict") is not None,
        "ema_rate": float(source_config.get("ema_rate", 0.0)),
        "use_ema_for_validation": bool(
            source_config.get("use_ema_for_validation", False)
        ),
        "use_ema_for_reverse": bool(
            source_config.get("use_ema_for_reverse", False)
        ),
        "model_used_for_reverse": model_used_for_reverse,
        "final_learning_rate": float(
            source_config.get("final_learning_rate", 0.0)
        ),
        "checkpoint_every_updates": int(
            source_config.get("checkpoint_every_updates", 0)
        ),
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
            "reverse_noise_path": str(reverse_noise_path),
            "reverse_noise_pool_steps": MAX_REVERSE_NOISE_STEPS,
            "reverse_noise_coupling": reverse_noise_coupling,
            "reverse_noise_coupling_exact": reverse_noise_coupling_exact,
            "replay_original_reverse_artifacts": bool(
                args.replay_original_reverse_artifacts
            ),
            "resolved_device": device,
            "beta_schedule": str(
                source_config.get("beta_schedule", "legacy-unit")
            ),
            "beta_0": float(source_config.get("beta_0", 0.001)),
            "beta_f": float(source_config.get("beta_f", 5.0)),
            "beta_t0": float(source_config.get("beta_t0", 0.0)),
            "beta_tf": float(source_config.get("beta_tf", 1.0)),
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

    if args.teacher_dataset_only:
        run_teacher_dataset_shard(args, output_dir=output_dir, log=log)
        return
    if args.skip_teacher_generation and not args.skip_training:
        raise ValueError("--skip-teacher-generation requires --skip-training")
    if args.skip_training:
        run_saved_model_evaluation(args, output_dir=output_dir, log=log)
        return
    if args.teacher is None:
        raise ValueError("--teacher is required unless --skip-training is used")

    dtype = to_dtype(args.dtype)
    device = resolve_device(args.device)
    beta_schedule = beta_schedule_from_args(args)
    validate_beta_schedule_time_range(
        beta_schedule,
        minimum_time=args.minimum_time,
        maximum_time=args.maximum_time,
    )

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
    initial_points_path = output_dir / "teacher_initial_points.pt"
    torch.save(
        {
            "train_initial_points": train_initial.detach().cpu(),
            "validation_initial_points": validation_initial.detach().cpu(),
        },
        initial_points_path,
    )

    time_samples_path = output_dir / "time_samples.pt"
    validation_time_samples_path = output_dir / "validation_time_samples.pt"
    noise_samples_path = output_dir / "teacher_noises.pt"
    if args.teacher_noises_path is not None:
        sibling_noise = args.teacher_noises_path
    elif args.teacher != "heat":
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

    if args.teacher_noises_path is not None and not sibling_noise.exists():
        raise FileNotFoundError(f"missing teacher noises: {sibling_noise}")
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

    if args.prepare_teacher_inputs_only:
        log(
            "saved teacher inputs only: "
            f"initial_points={initial_points_path.name} "
            f"times={time_samples_path.name}/{validation_time_samples_path.name} "
            f"noises={noise_samples_path.name}"
        )
        return

    teacher_started = time.perf_counter()
    if args.teacher == "malliavin" and args.teacher_implementation == "batched":
        train_dataset, _, train_effective_teacher_batch_sizes = (
            build_malliavin_teacher_dataset_batched(
                initial_points=train_initial,
                times=train_times,
                noises=train_noises,
                batch_size=args.teacher_batch_size,
                covariance_regularization=args.covariance_regularization,
                beta_schedule=beta_schedule,
            )
        )
        validation_dataset, _, validation_effective_teacher_batch_sizes = (
            build_malliavin_teacher_dataset_batched(
                initial_points=validation_initial,
                times=validation_times,
                noises=validation_noises,
                batch_size=args.teacher_batch_size,
                covariance_regularization=args.covariance_regularization,
                beta_schedule=beta_schedule,
            )
        )
    else:
        train_dataset = build_teacher_dataset(
            initial_points=train_initial,
            times=train_times,
            noises=train_noises,
            teacher=args.teacher,
            covariance_regularization=args.covariance_regularization,
            heat_terms=args.heat_terms,
            beta_schedule=beta_schedule,
        )
        validation_dataset = build_teacher_dataset(
            initial_points=validation_initial,
            times=validation_times,
            noises=validation_noises,
            teacher=args.teacher,
            covariance_regularization=args.covariance_regularization,
            heat_terms=args.heat_terms,
            beta_schedule=beta_schedule,
        )
        train_effective_teacher_batch_sizes = [1] * args.train_size
        validation_effective_teacher_batch_sizes = [1] * args.validation_size
    teacher_generation_seconds = time.perf_counter() - teacher_started

    train_started = time.perf_counter()
    checkpoint_callback = make_training_checkpoint_callback(
        output_dir=output_dir,
        args=args,
    )
    if args.score_parameterization == "upstream_scaled_score":
        if not isinstance(beta_schedule, LinearBetaSchedule):
            raise ValueError(
                "upstream_scaled_score training requires --beta-schedule linear"
            )
        model, history, training_state = train_s2_upstream_style_score_model(
            train_dataset,
            beta_schedule=beta_schedule,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            hidden=args.hidden,
            n_blocks=args.n_blocks,
            num_frequencies=args.num_frequencies,
            device=device,
            return_history=True,
            training_unit=args.training_unit,
            updates=args.updates,
            warmup_updates=args.warmup_updates,
            lr_scheduler=args.lr_scheduler,
            ema_rate=args.ema_rate,
            checkpoint_every_updates=args.checkpoint_every_updates,
            checkpoint_callback=checkpoint_callback,
            return_training_state=True,
        )
        training_path = "upstream_scaled_score"
    elif args.teacher == "malliavin":
        model, history, training_state = train_s2_marginal_score(
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
            training_unit=args.training_unit,
            updates=args.updates,
            warmup_updates=args.warmup_updates,
            lr_scheduler=args.lr_scheduler,
            ema_rate=args.ema_rate,
            checkpoint_every_updates=args.checkpoint_every_updates,
            checkpoint_callback=checkpoint_callback,
            return_training_state=True,
        )
        training_path = "marginal_skorokhod"
    else:
        model, history, training_state = train_s2_score_model(
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
            training_unit=args.training_unit,
            updates=args.updates,
            warmup_updates=args.warmup_updates,
            lr_scheduler=args.lr_scheduler,
            ema_rate=args.ema_rate,
            checkpoint_every_updates=args.checkpoint_every_updates,
            checkpoint_callback=checkpoint_callback,
            return_training_state=True,
        )
        training_path = "direct_score"
    training_seconds = time.perf_counter() - train_started
    ema_model = training_state["ema_model"]

    initial_train_loss = float(history.get("initial_train_loss", float("nan")))
    final_train_loss = float(history.get("final_train_loss", float("nan")))
    best_train_loss = float(history.get("best_train_loss", float("nan")))

    (
        validation_loss,
        validation_loss_online,
        validation_loss_ema,
        validation_model_source,
    ) = evaluate_selected_validation_loss(
        model,
        ema_model,
        validation_dataset,
        teacher=args.teacher,
        use_ema=args.use_ema_for_validation,
    )
    reverse_model, model_used_for_reverse = select_online_or_ema_model(
        model,
        ema_model,
        use_ema=args.use_ema_for_reverse,
        purpose="reverse sampling",
    )

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
        build_score_fn(reverse_model),
        terminal_time=args.maximum_time,
        n_steps=args.reverse_steps,
        standard_noise=reverse_noise,
        minimum_forward_time=args.minimum_time,
        beta_schedule=beta_schedule,
    )
    reverse_sampling_seconds = time.perf_counter() - reverse_started

    generated_cpu = generated.detach().cpu()
    train_dataset_cpu = {key: value.detach().cpu() for key, value in train_dataset.items()}
    validation_dataset_cpu = {key: value.detach().cpu() for key, value in validation_dataset.items()}

    checkpoint_payload = build_model_checkpoint_payload(
        selected_model=reverse_model,
        online_model=model,
        ema_model=ema_model,
        model_source=model_used_for_reverse,
        teacher=args.teacher,
        training_path=training_path,
        hidden=args.hidden,
        n_blocks=args.n_blocks,
        num_frequencies=args.num_frequencies,
        dtype=args.dtype,
        training_state=training_state,
        training_unit=args.training_unit,
        requested_epochs=args.epochs,
        requested_updates=args.updates,
        base_learning_rate=args.learning_rate,
        warmup_updates=args.warmup_updates,
        lr_scheduler=args.lr_scheduler,
        ema_rate=args.ema_rate,
        use_ema_for_validation=args.use_ema_for_validation,
        use_ema_for_reverse=args.use_ema_for_reverse,
        checkpoint_every_updates=args.checkpoint_every_updates,
        beta_schedule=args.beta_schedule,
        beta_0=args.beta_0,
        beta_f=args.beta_f,
        beta_t0=args.beta_t0,
        beta_tf=args.beta_tf,
    )
    torch.save(checkpoint_payload, output_dir / "model.pt")
    training_run_metadata = {
        "training_unit": args.training_unit,
        "requested_training_unit": args.training_unit,
        "requested_epochs": args.epochs,
        "requested_updates": args.updates,
        "actual_optimizer_updates": training_state["actual_optimizer_updates"],
        "updates_per_epoch": training_state["updates_per_epoch"],
        "effective_epochs": training_state["effective_epochs"],
        "base_learning_rate": args.learning_rate,
        "warmup_updates": args.warmup_updates,
        "lr_scheduler": args.lr_scheduler,
        "initial_learning_rate": training_state["initial_learning_rate"],
        "peak_learning_rate": training_state["peak_learning_rate"],
        "final_learning_rate": training_state["final_learning_rate"],
        "learning_rate_trace": training_state["learning_rate_trace"],
        "ema_enabled": ema_model is not None,
        "ema_rate": args.ema_rate,
        "use_ema_for_validation": args.use_ema_for_validation,
        "use_ema_for_reverse": args.use_ema_for_reverse,
        "model_used_for_reverse": model_used_for_reverse,
        "checkpoint_every_updates": args.checkpoint_every_updates,
    }
    run_config.update(training_run_metadata)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2, default=str)
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
    diagnostic_target_key = diagnostic_target_key_for_teacher(
        args.teacher,
        args.score_parameterization,
    )
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
        "score_parameterization": args.score_parameterization,
        "initial_train_loss": initial_train_loss,
        "final_train_loss": final_train_loss,
        "best_train_loss": best_train_loss,
        "validation_loss": validation_loss,
        "final_train_loss_online": final_train_loss,
        "best_train_loss_online": best_train_loss,
        "validation_loss_online": validation_loss_online,
        "validation_loss_ema": validation_loss_ema,
        "teacher_generation_seconds": teacher_generation_seconds,
        "teacher_implementation": args.teacher_implementation,
        "requested_teacher_batch_size": args.teacher_batch_size,
        "train_teacher_batching": {
            "minimum_batch_size": min(train_effective_teacher_batch_sizes),
            "maximum_batch_size": max(train_effective_teacher_batch_sizes),
            "n_chunks": len(train_effective_teacher_batch_sizes),
        },
        "validation_teacher_batching": {
            "minimum_batch_size": min(validation_effective_teacher_batch_sizes),
            "maximum_batch_size": max(validation_effective_teacher_batch_sizes),
            "n_chunks": len(validation_effective_teacher_batch_sizes),
        },
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
        "beta_schedule": args.beta_schedule,
        "beta_0": args.beta_0,
        "beta_f": args.beta_f,
        "beta_t0": args.beta_t0,
        "beta_tf": args.beta_tf,
        **training_run_metadata,
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
