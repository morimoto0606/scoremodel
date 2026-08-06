#!/usr/bin/env python3
"""Diagnose Heat checkpoint score definitions without reverse sampling.

The report separates output normalization, diffusion-score scaling, tangent
projection, and the antipodal Earthquake coordinate convention used by the
Upstream training code.
"""

from __future__ import annotations

import argparse
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
        build_model_from_training_checkpoint_with_normalization_trace,
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
        build_model_from_training_checkpoint_with_normalization_trace,
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
    "results/earthquake_teacher_comparison/score_definition_diagnostic"
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


def _summary(values: Tensor) -> dict[str, float]:
    values = values.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
    if values.numel() == 0:
        raise ValueError("cannot summarize an empty tensor")
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def project_s2_tangent(points: Tensor, vectors: Tensor) -> Tensor:
    if points.shape != vectors.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points and vectors must have the same [sample, 3] shape")
    return vectors - torch.sum(points * vectors, dim=1, keepdim=True) * points


def field_diagnostics(points: Tensor, field: Tensor) -> dict[str, object]:
    """Describe field magnitude and radial/tangent residual at each base point."""

    if points.shape != field.shape or points.ndim != 2:
        raise ValueError("points and field must have the same [sample, dimension] shape")
    point_norm = torch.linalg.vector_norm(field, dim=1)
    radial = torch.sum(points * field, dim=1)
    return {
        "l2_norm": _summary(point_norm),
        "global_l2_norm": float(torch.linalg.vector_norm(field.to(torch.float64))),
        "x_dot_score": _summary(radial),
        "absolute_x_dot_score": _summary(torch.abs(radial)),
    }


def pairwise_diagnostics(
    reference: Tensor,
    candidate: Tensor,
    *,
    epsilon: float = METRIC_EPSILON,
) -> dict[str, object]:
    """Compare two vector fields and estimate candidate/reference scale."""

    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("fields must have the same [sample, dimension] shape")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    reference = reference.to(torch.float64)
    candidate = candidate.to(torch.float64)
    reference_point_norm = torch.linalg.vector_norm(reference, dim=1)
    candidate_point_norm = torch.linalg.vector_norm(candidate, dim=1)
    cosine = torch.sum(reference * candidate, dim=1) / (
        reference_point_norm * candidate_point_norm
    ).clamp_min(epsilon)
    cosine = cosine.clamp(-1.0, 1.0)
    reference_l2 = torch.linalg.vector_norm(reference)
    candidate_l2 = torch.linalg.vector_norm(candidate)
    relative_l2 = torch.linalg.vector_norm(reference - candidate) / (
        reference_l2 + epsilon
    )
    # alpha minimizes ||candidate - alpha * reference||_2.
    scale = torch.sum(reference * candidate) / (
        torch.sum(reference * reference) + epsilon
    )
    scaled_residual = torch.linalg.vector_norm(candidate - scale * reference) / (
        candidate_l2 + epsilon
    )
    return {
        "cosine": _summary(cosine),
        "relative_l2_error": float(relative_l2),
        "norm_ratio": float(candidate_l2 / (reference_l2 + epsilon)),
        "least_squares_coordinate_scaling_factor": float(scale),
        "residual_after_scalar_fit": float(scaled_residual),
    }


def _normalization_buffers(normalized_model: object) -> dict[str, object]:
    names = ("x_mean", "x_std", "t_mean", "t_std", "y_mean", "y_std")
    return {
        name: getattr(normalized_model, name).detach().cpu().tolist() for name in names
    }


def _ext_forward_stages(
    normalized_model: object,
    times: Tensor,
    points: Tensor,
) -> dict[str, Tensor]:
    """Expose the exact inner-network and normalization-wrapper stages."""

    time_columns = times[:, None] if times.ndim == 1 else times
    normalized_time = (
        (time_columns - normalized_model.t_mean)
        / normalized_model.t_std.clamp_min(1e-6)
    ).squeeze(-1)
    normalized_points = (
        (points - normalized_model.x_mean)
        / normalized_model.x_std.clamp_min(1e-6)
    )
    raw_model_output = normalized_model.net(normalized_time, normalized_points)
    wrapper_output = normalized_model(times, points)
    expected_wrapper_output = (
        raw_model_output * normalized_model.y_std + normalized_model.y_mean
    )
    if not torch.allclose(wrapper_output, expected_wrapper_output, rtol=0.0, atol=0.0):
        raise RuntimeError("ext wrapper output does not match its explicit normalization path")
    return {
        "normalized_time": normalized_time,
        "normalized_points": normalized_points,
        "raw_model_output": raw_model_output,
        "wrapper_output": wrapper_output,
        "tangent_projected_output": project_s2_tangent(points, wrapper_output),
    }


def diagnose_score_definitions(
    points: Tensor,
    evaluation_times: Iterable[float],
    upstream_raw_output_fn: FieldFunction,
    normalized_ext_model: object,
    *,
    epsilon: float = METRIC_EPSILON,
) -> list[dict[str, object]]:
    """Run all forward-only score-definition diagnostics."""

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [sample, 3]")
    schedule = heat_beta_schedule()
    antipodal_points = -points
    results: list[dict[str, object]] = []
    with torch.no_grad():
        for time_value in evaluation_times:
            times = torch.full(
                (points.shape[0],),
                float(time_value),
                dtype=points.dtype,
                device=points.device,
            )
            sigma = upstream_score_standard_deviation(times, schedule)
            sigma_column = sigma[:, None]

            ext = _ext_forward_stages(normalized_ext_model, times, points)
            ext_at_antipode = _ext_forward_stages(
                normalized_ext_model, times, antipodal_points
            )
            # Pull the field at the antipode back to standard-earth coordinates.
            ext_antipodal_back = -ext_at_antipode["wrapper_output"]
            ext_antipodal_back_projected = project_s2_tangent(
                points, ext_antipodal_back
            )

            upstream_raw_same = upstream_raw_output_fn(times, points)
            upstream_effective_same = upstream_raw_same / sigma_column
            upstream_raw_native = upstream_raw_output_fn(times, antipodal_points)
            # Upstream native -> standard-earth point and tangent-vector transform.
            upstream_raw_aligned = -upstream_raw_native
            upstream_effective_aligned = upstream_raw_aligned / sigma_column

            ext_wrapper = ext["wrapper_output"]
            ext_projected = ext["tangent_projected_output"]
            up_effective_same_projected = project_s2_tangent(
                points, upstream_effective_same
            )
            up_effective_aligned_projected = project_s2_tangent(
                points, upstream_effective_aligned
            )

            ext_fields = {
                "raw_model_output": field_diagnostics(
                    points, ext["raw_model_output"]
                ),
                "wrapper_output": field_diagnostics(points, ext_wrapper),
                "tangent_projected_output": field_diagnostics(
                    points, ext_projected
                ),
                "coordinate_transform": {
                    "input_before": "standard-earth x",
                    "input_after": "upstream native -x",
                    "vector_pullback": "v_standard = -v_upstream",
                    "wrapper_output_at_minus_x_pulled_back": field_diagnostics(
                        points, ext_antipodal_back
                    ),
                    "pulled_back_tangent_projection": field_diagnostics(
                        points, ext_antipodal_back_projected
                    ),
                },
                "pairwise_cosine_and_scale": {
                    "raw_vs_wrapper": pairwise_diagnostics(
                        ext["raw_model_output"], ext_wrapper, epsilon=epsilon
                    ),
                    "wrapper_vs_tangent_projection": pairwise_diagnostics(
                        ext_wrapper, ext_projected, epsilon=epsilon
                    ),
                    "wrapper_x_vs_pulled_back_wrapper_minus_x": pairwise_diagnostics(
                        ext_wrapper, ext_antipodal_back, epsilon=epsilon
                    ),
                    "wrapper_vs_sigma_times_wrapper": pairwise_diagnostics(
                        ext_wrapper, sigma_column * ext_wrapper, epsilon=epsilon
                    ),
                },
                "normalized_model_input": {
                    "time": _summary(ext["normalized_time"]),
                    "xyz_component_values": _summary(ext["normalized_points"]),
                },
            }

            upstream_fields = {
                "same_numeric_xyz": {
                    "raw_output_N_up": field_diagnostics(points, upstream_raw_same),
                    "effective_score_N_up_over_sigma": field_diagnostics(
                        points, upstream_effective_same
                    ),
                    "raw_vs_effective": pairwise_diagnostics(
                        upstream_raw_same, upstream_effective_same, epsilon=epsilon
                    ),
                },
                "coordinate_aligned": {
                    "native_input": "-x_standard",
                    "output_transform": "-N_up(-x_standard)",
                    "raw_output_in_standard_coordinates": field_diagnostics(
                        points, upstream_raw_aligned
                    ),
                    "effective_score_in_standard_coordinates": field_diagnostics(
                        points, upstream_effective_aligned
                    ),
                },
            }

            def comparisons(up_raw: Tensor, up_effective: Tensor) -> dict[str, object]:
                up_projected = project_s2_tangent(points, up_effective)
                return {
                    "raw_output": {
                        "N_up_vs_sigma_times_ext_wrapper": pairwise_diagnostics(
                            up_raw, sigma_column * ext_wrapper, epsilon=epsilon
                        ),
                        "N_up_vs_ext_wrapper_no_sigma": pairwise_diagnostics(
                            up_raw, ext_wrapper, epsilon=epsilon
                        ),
                    },
                    "effective_score": {
                        "N_up_over_sigma_vs_ext_wrapper": pairwise_diagnostics(
                            up_effective, ext_wrapper, epsilon=epsilon
                        ),
                        "N_up_over_sigma_vs_ext_wrapper_over_sigma": pairwise_diagnostics(
                            up_effective, ext_wrapper / sigma_column, epsilon=epsilon
                        ),
                    },
                    "tangent_projection": {
                        "projected_upstream_vs_projected_ext": pairwise_diagnostics(
                            up_projected, ext_projected, epsilon=epsilon
                        ),
                        "upstream_projection_change": pairwise_diagnostics(
                            up_effective, up_projected, epsilon=epsilon
                        ),
                        "ext_projection_change": pairwise_diagnostics(
                            ext_wrapper, ext_projected, epsilon=epsilon
                        ),
                    },
                }

            results.append(
                {
                    "t": float(time_value),
                    "sigma": float(sigma[0]),
                    "ext_checkpoint": ext_fields,
                    "upstream_checkpoint": upstream_fields,
                    "comparisons": {
                        "same_numeric_xyz": comparisons(
                            upstream_raw_same, upstream_effective_same
                        ),
                        "coordinate_aligned": comparisons(
                            upstream_raw_aligned, upstream_effective_aligned
                        ),
                    },
                }
            )
    return results


def _mean_metric(
    results: list[dict[str, object]],
    coordinate_policy: str,
    section: str,
    comparison: str,
    metric: str,
) -> float:
    values = [
        row["comparisons"][coordinate_policy][section][comparison][metric]
        for row in results
    ]
    return float(sum(values) / len(values))


def infer_parameterization(results: list[dict[str, object]]) -> dict[str, object]:
    """Rank effective-score and sigma-score hypotheses over all times."""

    policies = ("same_numeric_xyz", "coordinate_aligned")
    report: dict[str, object] = {}
    for policy in policies:
        effective_error = _mean_metric(
            results,
            policy,
            "raw_output",
            "N_up_vs_sigma_times_ext_wrapper",
            "relative_l2_error",
        )
        sigma_score_error = _mean_metric(
            results,
            policy,
            "raw_output",
            "N_up_vs_ext_wrapper_no_sigma",
            "relative_l2_error",
        )
        report[policy] = {
            "ext_is_effective_score_mean_raw_equivalence_error": effective_error,
            "ext_is_sigma_times_score_mean_raw_equivalence_error": sigma_score_error,
            "lower_error_hypothesis": (
                "ext_output_is_effective_score"
                if effective_error < sigma_score_error
                else "ext_output_is_sigma_times_score"
            ),
            "interpretation_warning": (
                "This is empirical checkpoint evidence; architecture and training-target "
                "differences can prevent either error from being small."
            ),
        }
    return report


def save_diagnostic(
    output_dir: Path,
    *,
    upstream_checkpoint: Path | str,
    ext_checkpoint: Path | str,
    data_path: Path | str,
    sample_count: int,
    sample_seed: int,
    results: list[dict[str, object]],
    normalization_buffers: dict[str, object],
    normalization_trace: dict[str, object] | None = None,
    upstream_metadata: dict[str, object] | None = None,
) -> Path:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "purpose": "diagnose Upstream/ext Heat score-definition mismatch",
        "inference_only": True,
        "reverse_sampler_used": False,
        "upstream_checkpoint": str(upstream_checkpoint),
        "ext_checkpoint": str(ext_checkpoint),
        "data_path": str(data_path),
        "sample_count": int(sample_count),
        "sample_seed": int(sample_seed),
        "evaluation_times": [float(row["t"]) for row in results],
        "coordinate_definition": {
            "ext_training_coordinates": "standard-earth xyz",
            "upstream_training_coordinates": "antipodal xyz = -standard-earth xyz",
            "coordinate_norm_scaling_factor": 1.0,
            "coordinate_sign": -1.0,
            "tangent_vector_transform": "v_upstream = -v_standard",
            "same_numeric_xyz": "both networks receive x_standard",
            "coordinate_aligned": "ext receives x_standard; Upstream receives -x_standard and its output is negated",
        },
        "ext_normalization_buffers": normalization_buffers,
        "parameterization_evidence": infer_parameterization(results),
        "results": results,
    }
    if normalization_trace is not None:
        payload["ext_normalization_load_trace"] = normalization_trace
    if upstream_metadata is not None:
        payload["upstream_checkpoint_metadata"] = upstream_metadata
    output_path = output_dir / "score_definition_diagnostic.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return output_path


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

    upstream_fn, upstream_dtype, upstream_metadata = _build_raw_ema_network(
        upstream_checkpoint
    )
    if upstream_dtype != str(run_config["dtype"]):
        raise ValueError(
            "checkpoint dtype mismatch: "
            f"Upstream={upstream_dtype}, ext={run_config['dtype']}"
        )
    ext_model, normalization_trace, normalized_ext_model, _ = (
        build_model_from_training_checkpoint_with_normalization_trace(
            ext_checkpoint, device=device
        )
    )
    require_exact_checkpoint_state(ext_model, ext_checkpoint)
    points = _sample_earthquake_points(
        args.data_path.expanduser().resolve(),
        sample_count=args.sample_count,
        seed=args.sample_seed,
        dtype=dtype,
        device=device,
    )
    results = diagnose_score_definitions(
        points,
        EVALUATION_TIMES,
        upstream_fn,
        normalized_ext_model,
    )
    output_path = save_diagnostic(
        args.output_dir,
        upstream_checkpoint=upstream_checkpoint,
        ext_checkpoint=ext_checkpoint,
        data_path=args.data_path.expanduser().resolve(),
        sample_count=points.shape[0],
        sample_seed=args.sample_seed,
        results=results,
        normalization_buffers=_normalization_buffers(normalized_ext_model),
        normalization_trace=normalization_trace,
        upstream_metadata=upstream_metadata,
    )
    print(f"saved {output_path}")
    evidence = infer_parameterization(results)
    for policy, values in evidence.items():
        print(
            f"{policy}: {values['lower_error_hypothesis']} "
            f"effective_error={values['ext_is_effective_score_mean_raw_equivalence_error']:.6g} "
            f"sigma_score_error={values['ext_is_sigma_times_score_mean_raw_equivalence_error']:.6g}"
        )


if __name__ == "__main__":
    main()
