#!/usr/bin/env python3
"""Compare equivalent raw-output parameterizations of two Heat checkpoints.

This is an inference-only checkpoint comparison.  It evaluates the Upstream
EMA network output ``N_up(x, t)`` against ``sigma(t) * s_ext(x, t)`` on the
same Earthquake points.  It does not construct or call a reverse sampler.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable, Iterable

import torch

if __package__:
    from .compare_heat_checkpoint_scores import (
        DEFAULT_DATA_PATH,
        DEFAULT_EXT_CHECKPOINT,
        DEFAULT_UPSTREAM_CHECKPOINT,
        EVALUATION_TIMES,
        METRIC_EPSILON,
        SAMPLE_COUNT,
        _sample_earthquake_points,
        heat_beta_schedule,
    )
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
    from compare_heat_checkpoint_scores import (
        DEFAULT_DATA_PATH,
        DEFAULT_EXT_CHECKPOINT,
        DEFAULT_UPSTREAM_CHECKPOINT,
        EVALUATION_TIMES,
        METRIC_EPSILON,
        SAMPLE_COUNT,
        _sample_earthquake_points,
        heat_beta_schedule,
    )
    from experiment_earthquake_teacher_compare_smoke import (
        build_model_from_training_checkpoint,
        build_score_fn,
        load_run_config_for_model,
        require_exact_checkpoint_state,
        resolve_device,
        to_dtype,
    )
    from run_upstream_checkpoint_with_ext_sampler import _build_raw_ema_network

from scoremodel_ext.manifold.s2_reverse_diagnostics import (
    upstream_score_standard_deviation,
)


Tensor = torch.Tensor
FieldFunction = Callable[[Tensor, Tensor], Tensor]

DEFAULT_OUTPUT_DIR = Path(
    "results/earthquake_teacher_comparison/raw_output_comparison"
)


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


def compute_raw_output_metrics(
    upstream_raw_output: Tensor,
    scaled_ext_score: Tensor,
    *,
    epsilon: float = METRIC_EPSILON,
) -> dict[str, float]:
    """Compare fields using per-point cosine and global relative L2 metrics."""

    if (
        upstream_raw_output.shape != scaled_ext_score.shape
        or upstream_raw_output.ndim != 2
    ):
        raise ValueError("fields must have the same [sample, dimension] shape")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    upstream = upstream_raw_output.to(dtype=torch.float64)
    ext_scaled = scaled_ext_score.to(dtype=torch.float64)
    upstream_point_norm = torch.linalg.vector_norm(upstream, dim=1)
    ext_point_norm = torch.linalg.vector_norm(ext_scaled, dim=1)
    cosine = torch.sum(upstream * ext_scaled, dim=1) / (
        upstream_point_norm * ext_point_norm
    ).clamp_min(epsilon)
    cosine = cosine.clamp(-1.0, 1.0)

    upstream_l2 = torch.linalg.vector_norm(upstream)
    ext_l2 = torch.linalg.vector_norm(ext_scaled)
    difference_l2 = torch.linalg.vector_norm(upstream - ext_scaled)
    return {
        "cosine_similarity": float(cosine.mean()),
        "cosine_similarity_std": float(cosine.std(unbiased=False)),
        "cosine_similarity_min": float(cosine.min()),
        "relative_l2_error": float(difference_l2 / (upstream_l2 + epsilon)),
        "norm_ratio": float(ext_l2 / (upstream_l2 + epsilon)),
        "upstream_raw_output_l2_norm": float(upstream_l2),
        "scaled_ext_score_l2_norm": float(ext_l2),
    }


def evaluate_raw_outputs(
    points: Tensor,
    evaluation_times: Iterable[float],
    upstream_raw_output_fn: FieldFunction,
    ext_effective_score_fn: FieldFunction,
    *,
    epsilon: float = METRIC_EPSILON,
) -> list[dict[str, float]]:
    """Evaluate ``N_up`` and ``sigma * s_ext`` at identical points and times."""

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [sample, 3]")
    schedule = heat_beta_schedule()
    results = []
    with torch.no_grad():
        for time_value in evaluation_times:
            times = torch.full(
                (points.shape[0],),
                float(time_value),
                dtype=points.dtype,
                device=points.device,
            )
            upstream_raw_output = upstream_raw_output_fn(times, points)
            ext_effective_score = ext_effective_score_fn(times, points)
            sigma = upstream_score_standard_deviation(times, schedule)
            scaled_ext_score = sigma[:, None] * ext_effective_score
            metrics = compute_raw_output_metrics(
                upstream_raw_output,
                scaled_ext_score,
                epsilon=epsilon,
            )
            results.append(
                {
                    "t": float(time_value),
                    "sigma": float(sigma[0]),
                    **metrics,
                }
            )
    return results


def save_raw_output_comparison(
    output_dir: Path,
    *,
    upstream_checkpoint: Path | str,
    ext_checkpoint: Path | str,
    sample_count: int,
    results: list[dict[str, float]],
    metadata: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "upstream_checkpoint": str(upstream_checkpoint),
        "ext_checkpoint": str(ext_checkpoint),
        "sample_count": int(sample_count),
        "comparison": "N_up(x,t) vs sigma(t) * ext_effective_score(x,t)",
        "results": results,
    }
    if metadata is not None:
        payload["metadata"] = metadata

    json_path = output_dir / "raw_output_comparison.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    csv_path = output_dir / "raw_output_comparison.csv"
    fieldnames = [
        "t",
        "sigma",
        "cosine_similarity",
        "cosine_similarity_std",
        "cosine_similarity_min",
        "relative_l2_error",
        "norm_ratio",
        "upstream_raw_output_l2_norm",
        "scaled_ext_score_l2_norm",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    return json_path, csv_path


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
    results = evaluate_raw_outputs(
        points,
        EVALUATION_TIMES,
        upstream_raw_output_fn,
        ext_effective_score_fn,
    )
    json_path, csv_path = save_raw_output_comparison(
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
            "upstream_quantity": "raw network output N_up(x,t)",
            "ext_quantity": "sqrt(1-exp(-tau(t))) * effective_score(x,t)",
            "beta_schedule": {
                "type": "linear",
                "beta_0": 0.001,
                "beta_f": 5.0,
                "t0": 0.0,
                "tf": 1.0,
            },
            "metric_epsilon": METRIC_EPSILON,
            "reverse_sampler_used": False,
            "upstream_checkpoint_metadata": upstream_metadata,
        },
    )
    print(f"saved {json_path}")
    print(f"saved {csv_path}")
    for row in results:
        print(
            f"t={row['t']:.3g} sigma={row['sigma']:.6g} "
            f"cosine={row['cosine_similarity']:.6g} "
            f"relative_l2_error={row['relative_l2_error']:.6g} "
            f"norm_ratio={row['norm_ratio']:.6g}"
        )


if __name__ == "__main__":
    main()
