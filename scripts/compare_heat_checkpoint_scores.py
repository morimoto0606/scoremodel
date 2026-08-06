#!/usr/bin/env python3
"""Directly compare Upstream and scoremodel_ext Heat score fields.

No reverse sampling or training is performed.  Both checkpoints are evaluated
on the same Earthquake points and physical times.  The Upstream EMA network's
raw output is converted to a score exactly once; the ext checkpoint output is
used directly as an effective score.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable, Iterable

import torch

if __package__:
    from .experiment_earthquake_teacher_compare_smoke import (
        build_model_from_training_checkpoint,
        build_score_fn,
        load_run_config_for_model,
        require_exact_checkpoint_state,
        resolve_device,
        to_dtype,
    )
    from .run_upstream_checkpoint_with_ext_sampler import _build_raw_ema_network
else:
    from experiment_earthquake_teacher_compare_smoke import (
        build_model_from_training_checkpoint,
        build_score_fn,
        load_run_config_for_model,
        require_exact_checkpoint_state,
        resolve_device,
        to_dtype,
    )
    from run_upstream_checkpoint_with_ext_sampler import _build_raw_ema_network

from scoremodel_ext.manifold.beta_schedule import LinearBetaSchedule
from scoremodel_ext.manifold.earthquake_adapter import load_earthquake_points
from scoremodel_ext.manifold.s2_reverse_diagnostics import (
    upstream_score_standard_deviation,
)


Tensor = torch.Tensor
ScoreFunction = Callable[[Tensor, Tensor], Tensor]

DEFAULT_UPSTREAM_CHECKPOINT = Path(
    "results/earthquake_teacher_comparison/upstream_heat/ckpt"
)
DEFAULT_EXT_CHECKPOINT = Path("results/earthquake_linear_beta_100k_ema_heat/model.pt")
DEFAULT_DATA_PATH = Path("upstream/riemannian-score-sde/data/quakes_all.csv")
DEFAULT_OUTPUT_DIR = Path(
    "results/earthquake_teacher_comparison/score_field_comparison"
)
EVALUATION_TIMES = (0.001, 0.01, 0.05, 0.1, 0.5, 0.9, 1.0)
SAMPLE_COUNT = 4096
METRIC_EPSILON = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream-checkpoint", type=Path, default=DEFAULT_UPSTREAM_CHECKPOINT
    )
    parser.add_argument("--ext-checkpoint", type=Path, default=DEFAULT_EXT_CHECKPOINT)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-count", type=int, default=SAMPLE_COUNT)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.sample_count < 1:
        parser.error("--sample-count must be positive")
    return args


def heat_beta_schedule() -> LinearBetaSchedule:
    return LinearBetaSchedule(beta_0=0.001, beta_f=5.0, t0=0.0, tf=1.0)


def compute_score_metrics(
    upstream_score: Tensor,
    ext_score: Tensor,
    *,
    epsilon: float = METRIC_EPSILON,
) -> dict[str, float]:
    """Return requested per-sample vector-field comparison summaries."""

    if upstream_score.shape != ext_score.shape or upstream_score.ndim != 2:
        raise ValueError("score tensors must have the same [sample, dimension] shape")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    upstream_score = upstream_score.to(dtype=torch.float64)
    ext_score = ext_score.to(dtype=torch.float64)
    upstream_norm = torch.linalg.vector_norm(upstream_score, dim=1)
    ext_norm = torch.linalg.vector_norm(ext_score, dim=1)
    cosine = torch.sum(upstream_score * ext_score, dim=1) / (
        upstream_norm * ext_norm
    ).clamp_min(epsilon)
    cosine = cosine.clamp(-1.0, 1.0)
    relative_error = torch.linalg.vector_norm(
        upstream_score - ext_score, dim=1
    ) / (upstream_norm + epsilon)
    norm_ratio = ext_norm / (upstream_norm + epsilon)
    return {
        "cosine_mean": float(cosine.mean()),
        "cosine_std": float(cosine.std(unbiased=False)),
        "cosine_min": float(cosine.min()),
        "relative_error_mean": float(relative_error.mean()),
        "relative_error_std": float(relative_error.std(unbiased=False)),
        "norm_ratio_mean": float(norm_ratio.mean()),
        "norm_ratio_std": float(norm_ratio.std(unbiased=False)),
    }


def evaluate_score_fields(
    points: Tensor,
    evaluation_times: Iterable[float],
    upstream_raw_output_fn: ScoreFunction,
    ext_effective_score_fn: ScoreFunction,
    *,
    beta_schedule: LinearBetaSchedule | None = None,
    epsilon: float = METRIC_EPSILON,
) -> list[dict[str, float]]:
    """Evaluate both checkpoint fields on identical numeric S2 coordinates."""

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [sample, 3]")
    schedule = heat_beta_schedule() if beta_schedule is None else beta_schedule
    results = []
    with torch.no_grad():
        for time_value in evaluation_times:
            time_batch = torch.full(
                (points.shape[0],),
                float(time_value),
                dtype=points.dtype,
                device=points.device,
            )
            upstream_network_output = upstream_raw_output_fn(time_batch, points)
            score_std = upstream_score_standard_deviation(time_batch, schedule)
            upstream_score = upstream_network_output / score_std[:, None]
            # The ext checkpoint already emits the effective score.  Do not
            # divide it by the Upstream standard deviation a second time.
            ext_score = ext_effective_score_fn(time_batch, points)
            metrics = compute_score_metrics(
                upstream_score,
                ext_score,
                epsilon=epsilon,
            )
            results.append({"t": float(time_value), **metrics})
    return results


def save_score_comparison(
    output_dir: Path,
    *,
    upstream_checkpoint: Path | str,
    ext_checkpoint: Path | str,
    sample_count: int,
    results: list[dict[str, float]],
    metadata: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    """Save the comparison in the requested JSON and CSV formats."""

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "upstream_checkpoint": str(upstream_checkpoint),
        "ext_checkpoint": str(ext_checkpoint),
        "sample_count": int(sample_count),
        "results": results,
    }
    if metadata is not None:
        payload["metadata"] = metadata
    json_path = output_dir / "score_comparison.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    csv_path = output_dir / "score_comparison.csv"
    fieldnames = [
        "t",
        "cosine_mean",
        "cosine_std",
        "cosine_min",
        "relative_error_mean",
        "relative_error_std",
        "norm_ratio_mean",
        "norm_ratio_std",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    return json_path, csv_path


def _sample_earthquake_points(
    path: Path,
    *,
    sample_count: int,
    seed: int,
    dtype: torch.dtype,
    device: str,
) -> Tensor:
    all_points = load_earthquake_points(path, dtype=dtype, device="cpu")
    if sample_count > all_points.shape[0]:
        raise ValueError(
            f"requested {sample_count} points but dataset has {all_points.shape[0]}"
        )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randperm(all_points.shape[0], generator=generator)[:sample_count]
    return all_points[indices].to(device=device)


def main() -> None:
    args = parse_args()
    upstream_checkpoint = args.upstream_checkpoint.expanduser().resolve()
    if not upstream_checkpoint.is_dir():
        raise FileNotFoundError(f"missing Upstream checkpoint: {upstream_checkpoint}")
    ext_checkpoint, run_config = load_run_config_for_model(args.ext_checkpoint)
    if str(run_config.get("teacher")) != "heat":
        raise ValueError("the ext checkpoint must be a Heat model")
    device = resolve_device(args.device)
    dtype = to_dtype(str(run_config["dtype"]))

    upstream_raw_output_fn, upstream_dtype, upstream_metadata = (
        _build_raw_ema_network(upstream_checkpoint)
    )
    if upstream_dtype != str(run_config["dtype"]):
        raise ValueError(
            "checkpoint dtype mismatch: "
            f"Upstream={upstream_dtype}, ext={run_config['dtype']}"
        )
    ext_model = build_model_from_training_checkpoint(ext_checkpoint, device=device)
    require_exact_checkpoint_state(ext_model, ext_checkpoint)
    ext_effective_score_fn = build_score_fn(ext_model)
    points = _sample_earthquake_points(
        args.data_path.expanduser().resolve(),
        sample_count=args.sample_count,
        seed=args.sample_seed,
        dtype=dtype,
        device=device,
    )
    results = evaluate_score_fields(
        points,
        EVALUATION_TIMES,
        upstream_raw_output_fn,
        ext_effective_score_fn,
    )
    json_path, csv_path = save_score_comparison(
        args.output_dir,
        upstream_checkpoint=upstream_checkpoint,
        ext_checkpoint=ext_checkpoint,
        sample_count=points.shape[0],
        results=results,
        metadata={
            "evaluation_times": list(EVALUATION_TIMES),
            "sample_seed": args.sample_seed,
            "data_path": str(args.data_path.expanduser().resolve()),
            "input_coordinate_policy": "identical standard-earth xyz passed to both models",
            "upstream_parameter_source": "params_ema",
            "upstream_score": "network_output / sqrt(1-exp(-tau(t)))",
            "ext_score": "model(x,t)",
            "ext_additional_std_division": False,
            "metric_epsilon": METRIC_EPSILON,
            "upstream_checkpoint_metadata": upstream_metadata,
        },
    )
    print(f"saved {json_path}")
    print(f"saved {csv_path}")
    for row in results:
        print(
            f"t={row['t']:.3g} cosine={row['cosine_mean']:.6g} "
            f"relative_error={row['relative_error_mean']:.6g} "
            f"norm_ratio={row['norm_ratio_mean']:.6g}"
        )


if __name__ == "__main__":
    main()
