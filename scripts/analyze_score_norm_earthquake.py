#!/usr/bin/env python3
"""Diagnose Earthquake score magnitude from a saved scoremodel_ext checkpoint.

The default path evaluates the restored model on saved validation points.  An
opt-in reverse diagnostic wraps the score function passed to ``s2_reverse_grw``
and records the score actually consumed at every reverse step.  No teacher data
or model parameters are generated or modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from experiment_earthquake_teacher_compare_smoke import (
    beta_schedule_from_run_config,
    build_model_from_training_checkpoint,
    build_score_fn,
    load_run_config_for_model,
    require_exact_checkpoint_state,
    resolve_device,
    to_dtype,
)
from scoremodel_ext.manifold.s2_malliavin import s2_reverse_grw


DEFAULT_TIME_GRID = (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
POINT_KEYS = (
    "validation_initial_points",
    "endpoint",
    "output_points",
    "generated_samples",
    "terminal_samples",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--points-path", type=Path, default=None)
    parser.add_argument(
        "--points-key",
        default=None,
        help="dictionary key for --points-path; inferred only for known keys",
    )
    parser.add_argument(
        "--time-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_TIME_GRID),
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-points", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--label", default=None)
    parser.add_argument(
        "--compare-report",
        type=Path,
        action="append",
        default=[],
        help="optional report with the same JSON schema to overlay in the plot",
    )
    parser.add_argument("--analyze-reverse", action="store_true")
    parser.add_argument("--terminal-samples-path", type=Path, default=None)
    parser.add_argument("--reverse-noise-path", type=Path, default=None)
    parser.add_argument("--reverse-steps", type=int, default=None)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.max_points < 0:
        parser.error("--max-points must be non-negative")
    if not args.time_grid:
        parser.error("--time-grid must not be empty")
    if args.reverse_steps is not None and args.reverse_steps < 1:
        parser.error("--reverse-steps must be positive")
    return args


def _load_serialized_value(path: Path):
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing artifact: {resolved}")
    if resolved.suffix == ".npy":
        return torch.from_numpy(np.load(resolved, allow_pickle=False))
    return torch.load(resolved, map_location="cpu")


def load_point_artifact(path: Path, *, key: str | None) -> torch.Tensor:
    value = _load_serialized_value(path)
    if isinstance(value, Mapping):
        if key is None:
            candidates = [candidate for candidate in POINT_KEYS if candidate in value]
            if len(candidates) != 1:
                raise ValueError(
                    f"cannot infer point key from {path}; candidates={candidates}. "
                    "Specify --points-key."
                )
            key = candidates[0]
        if key not in value:
            raise KeyError(f"point artifact {path} has no key {key!r}")
        value = value[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"point artifact must contain a tensor, got {type(value).__name__}")
    if value.ndim != 2 or value.shape[1] != 3 or value.shape[0] < 1:
        raise ValueError(f"points must have shape (n, 3), got {tuple(value.shape)}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("points contain non-finite values")
    norms = torch.linalg.vector_norm(value, dim=1)
    maximum_norm_error = torch.max(torch.abs(norms - 1.0))
    if bool(maximum_norm_error > 1e-4):
        raise ValueError(
            "points are not on the unit sphere: "
            f"maximum norm error={float(maximum_norm_error):.6g}"
        )
    return value


def _load_reverse_artifact(path: Path, *, key: str) -> torch.Tensor:
    value = _load_serialized_value(path)
    if isinstance(value, Mapping):
        if key not in value:
            raise KeyError(f"reverse artifact {path} has no key {key!r}")
        value = value[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"reverse artifact must contain a tensor, got {type(value).__name__}")
    return value


def _project_to_sphere_tangent(points: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    normalized = points / torch.linalg.vector_norm(points, dim=1, keepdim=True)
    return vectors - (vectors * normalized).sum(dim=1, keepdim=True) * normalized


def _summary(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().to(dtype=torch.float64, device="cpu")
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _cosine_summary(left: torch.Tensor, right: torch.Tensor) -> dict[str, float] | None:
    left_norm = torch.linalg.vector_norm(left, dim=1)
    right_norm = torch.linalg.vector_norm(right, dim=1)
    valid = (left_norm > 1e-12) & (right_norm > 1e-12)
    if not bool(valid.any()):
        return None
    cosine = F.cosine_similarity(left[valid], right[valid], dim=1, eps=1e-12)
    result = _summary(cosine)
    result["count"] = int(valid.sum().detach().cpu())
    return result


def _beta_values(schedule, time_batch: torch.Tensor) -> torch.Tensor:
    if schedule is None:
        return torch.ones_like(time_batch)
    return schedule.beta_t(time_batch)


def analyze_fixed_points(
    model,
    points: torch.Tensor,
    time_grid: Sequence[float],
    *,
    schedule,
    batch_size: int,
) -> list[dict]:
    rows = []
    previous_projected = None
    for time_value in time_grid:
        raw_parts = []
        projected_parts = []
        with torch.no_grad():
            for start in range(0, points.shape[0], batch_size):
                point_batch = points[start : start + batch_size]
                time_batch = torch.full(
                    (point_batch.shape[0],),
                    float(time_value),
                    dtype=points.dtype,
                    device=points.device,
                )
                raw = model(time_batch, point_batch)
                projected = _project_to_sphere_tangent(point_batch, raw)
                raw_parts.append(raw.detach())
                projected_parts.append(projected.detach())
        raw_score = torch.cat(raw_parts, dim=0)
        projected_score = torch.cat(projected_parts, dim=0)
        time_batch = torch.full(
            (points.shape[0],),
            float(time_value),
            dtype=points.dtype,
            device=points.device,
        )
        beta = _beta_values(schedule, time_batch)
        sigma_squared_score = beta[:, None] * projected_score
        row = {
            "time": float(time_value),
            "beta": float(beta[0].detach().cpu()),
            "raw_score_norm": _summary(torch.linalg.vector_norm(raw_score, dim=1)),
            "projected_score_norm": _summary(
                torch.linalg.vector_norm(projected_score, dim=1)
            ),
            "sigma_squared_projected_score_norm": _summary(
                torch.linalg.vector_norm(sigma_squared_score, dim=1)
            ),
            "raw_projected_cosine": _cosine_summary(raw_score, projected_score),
            "projected_cosine_to_previous_time": (
                None
                if previous_projected is None
                else _cosine_summary(projected_score, previous_projected)
            ),
        }
        rows.append(row)
        previous_projected = projected_score
    return rows


def analyze_reverse_path(
    model,
    terminal_samples: torch.Tensor,
    reverse_noise: torch.Tensor,
    *,
    run_config: Mapping[str, object],
    schedule,
    reverse_steps: int,
) -> tuple[list[dict], dict[str, float]]:
    terminal_time = float(run_config["maximum_time"])
    minimum_time = float(run_config["minimum_time"])
    schedule_t0 = 0.0 if schedule is None else float(schedule.t0)
    physical_dt = (terminal_time - schedule_t0) / reverse_steps
    base_score_fn = build_score_fn(model)
    rows = []

    def traced_score_fn(time_batch: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        raw_score = base_score_fn(time_batch, points)
        projected_score = _project_to_sphere_tangent(points, raw_score)
        beta = _beta_values(schedule, time_batch)
        step = len(rows)
        physical_time = terminal_time - step * physical_dt
        next_physical_time = terminal_time - (step + 1) * physical_dt
        delta_tau = (
            physical_dt
            if schedule is None
            else schedule.interval_brownian_time(next_physical_time, physical_time)
        )
        sigma_squared_score = beta[:, None] * projected_score
        drift_increment = float(delta_tau) * projected_score
        projected_noise = _project_to_sphere_tangent(points, reverse_noise[step])
        noise_increment = float(delta_tau) ** 0.5 * projected_noise
        drift_norm = torch.linalg.vector_norm(drift_increment, dim=1)
        noise_norm = torch.linalg.vector_norm(noise_increment, dim=1)
        rows.append(
            {
                "step": step,
                "physical_time": physical_time,
                "network_time": float(time_batch[0].detach().cpu()),
                "beta": float(beta[0].detach().cpu()),
                "delta_tau": float(delta_tau),
                "raw_score_norm": _summary(torch.linalg.vector_norm(raw_score, dim=1)),
                "projected_score_norm": _summary(
                    torch.linalg.vector_norm(projected_score, dim=1)
                ),
                "sigma_squared_projected_score_norm": _summary(
                    torch.linalg.vector_norm(sigma_squared_score, dim=1)
                ),
                "score_drift_increment_norm": _summary(
                    drift_norm
                ),
                "noise_increment_norm": _summary(noise_norm),
                "drift_to_noise_norm_ratio": _summary(
                    drift_norm / noise_norm.clamp_min(1e-12)
                ),
                "raw_projected_cosine": _cosine_summary(raw_score, projected_score),
            }
        )
        return raw_score

    with torch.no_grad():
        generated = s2_reverse_grw(
            terminal_samples,
            traced_score_fn,
            terminal_time=terminal_time,
            n_steps=reverse_steps,
            standard_noise=reverse_noise,
            minimum_forward_time=minimum_time,
            beta_schedule=schedule,
        )
    final_norm = torch.linalg.vector_norm(generated, dim=1)
    final_summary = {
        "mean_norm": float(final_norm.mean().detach().cpu()),
        "maximum_norm_error": float(torch.max(torch.abs(final_norm - 1.0)).detach().cpu()),
        "finite_rate": float(torch.isfinite(generated).all(dim=1).double().mean().cpu()),
    }
    return rows, final_summary


def _report_label(report: Mapping[str, object], fallback: str) -> str:
    value = report.get("label")
    return fallback if value is None else str(value)


def save_plot(reports: Sequence[Mapping[str, object]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=220)
    colors = {
        "raw_score_norm": "#4c78a8",
        "projected_score_norm": "#f58518",
        "sigma_squared_projected_score_norm": "#54a24b",
    }
    for report_index, report in enumerate(reports):
        rows = report["time_statistics"]
        times = [float(row["time"]) for row in rows]
        label = _report_label(report, f"report-{report_index + 1}")
        for metric, color in colors.items():
            means = [float(row[metric]["mean"]) for row in rows]
            linestyle = "-" if report_index == 0 else "--"
            axes[0].plot(
                times,
                means,
                marker="o",
                linestyle=linestyle,
                color=color,
                label=f"{label}: {metric}",
            )

        reverse_rows = report.get("reverse_statistics") or []
        if reverse_rows:
            reverse_times = [float(row["physical_time"]) for row in reverse_rows]
            effective = [
                float(row["sigma_squared_projected_score_norm"]["mean"])
                for row in reverse_rows
            ]
            increment = [
                float(row["score_drift_increment_norm"]["mean"])
                for row in reverse_rows
            ]
            noise_increment = [
                float(row["noise_increment_norm"]["mean"])
                for row in reverse_rows
            ]
            axes[1].plot(reverse_times, effective, label=f"{label}: beta*score")
            axes[1].plot(reverse_times, increment, label=f"{label}: delta_tau*score")
            axes[1].plot(
                reverse_times,
                noise_increment,
                linestyle=":",
                label=f"{label}: sqrt(delta_tau)*noise",
            )

    axes[0].set_xlabel("Physical time t")
    axes[0].set_ylabel("Mean score norm")
    axes[0].set_title("Score magnitude on fixed points")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("Physical time t (reverse order)")
    axes[1].set_ylabel("Mean norm")
    axes[1].set_title("Effective reverse score drift")
    axes[1].grid(alpha=0.25)
    if any(report.get("reverse_statistics") for report in reports):
        axes[1].legend(fontsize=8)
    else:
        axes[1].text(
            0.5,
            0.5,
            "Run with --analyze-reverse\nto populate sampler diagnostics",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    model_path, run_config = load_run_config_for_model(args.model_path)
    output_dir = (
        model_path.parent if args.output_dir is None else args.output_dir.expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    dtype = to_dtype(str(run_config["dtype"]))
    schedule = beta_schedule_from_run_config(run_config)
    model = build_model_from_training_checkpoint(model_path, device=device)
    require_exact_checkpoint_state(model, model_path)

    points_path = (
        model_path.parent / "target_samples.pt"
        if args.points_path is None
        else args.points_path.expanduser().resolve()
    )
    points = load_point_artifact(points_path, key=args.points_key).to(
        device=device,
        dtype=dtype,
    )
    if args.max_points > 0:
        points = points[: args.max_points]
    minimum_time = float(run_config["minimum_time"])
    maximum_time = float(run_config["maximum_time"])
    invalid_times = [
        value for value in args.time_grid if value < minimum_time or value > maximum_time
    ]
    if invalid_times:
        raise ValueError(
            f"time grid is outside [{minimum_time}, {maximum_time}]: {invalid_times}"
        )

    time_statistics = analyze_fixed_points(
        model,
        points,
        args.time_grid,
        schedule=schedule,
        batch_size=args.batch_size,
    )
    checkpoint = torch.load(model_path, map_location="cpu")
    report = {
        "schema_version": 1,
        "backend": "scoremodel_ext-pytorch",
        "label": args.label or f"scoremodel_ext-{run_config['teacher']}",
        "model_path": str(model_path),
        "teacher": str(run_config["teacher"]),
        "model_source": str(checkpoint.get("model_source", "unknown")),
        "dtype": str(run_config["dtype"]),
        "device": device,
        "points_path": str(points_path),
        "point_count": int(points.shape[0]),
        "time_grid": [float(value) for value in args.time_grid],
        "beta_schedule": str(run_config.get("beta_schedule", "legacy-unit")),
        "time_statistics": time_statistics,
        "reverse_statistics": None,
        "reverse_final_samples": None,
    }

    if args.analyze_reverse:
        reverse_steps = (
            int(run_config["reverse_steps"])
            if args.reverse_steps is None
            else args.reverse_steps
        )
        terminal_path = (
            model_path.parent / "terminal_samples.pt"
            if args.terminal_samples_path is None
            else args.terminal_samples_path.expanduser().resolve()
        )
        noise_path = (
            model_path.parent / "reverse_noise.pt"
            if args.reverse_noise_path is None
            else args.reverse_noise_path.expanduser().resolve()
        )
        terminal_samples = load_point_artifact(terminal_path, key="terminal_samples").to(
            device=device,
            dtype=dtype,
        )
        reverse_noise = _load_reverse_artifact(noise_path, key="reverse_noise").to(
            device=device,
            dtype=dtype,
        )
        if args.max_points > 0:
            terminal_samples = terminal_samples[: args.max_points]
            reverse_noise = reverse_noise[:, : args.max_points]
        expected_noise_shape = (reverse_steps, terminal_samples.shape[0], 3)
        if tuple(reverse_noise.shape) != expected_noise_shape:
            raise ValueError(
                f"reverse noise must have shape {expected_noise_shape}, "
                f"got {tuple(reverse_noise.shape)}"
            )
        reverse_statistics, final_summary = analyze_reverse_path(
            model,
            terminal_samples,
            reverse_noise,
            run_config=run_config,
            schedule=schedule,
            reverse_steps=reverse_steps,
        )
        report["reverse_statistics"] = reverse_statistics
        report["reverse_final_samples"] = final_summary
        report["reverse_steps"] = reverse_steps
        report["terminal_samples_path"] = str(terminal_path)
        report["reverse_noise_path"] = str(noise_path)

    report_path = output_dir / "score_norm_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    reports = [report]
    for comparison_path in args.compare_report:
        with comparison_path.expanduser().resolve().open("r", encoding="utf-8") as handle:
            comparison = json.load(handle)
        if not isinstance(comparison, dict) or "time_statistics" not in comparison:
            raise ValueError(f"invalid comparison report: {comparison_path}")
        reports.append(comparison)
    plot_path = output_dir / "score_norm_plot.png"
    save_plot(reports, plot_path)
    print(f"saved {report_path}")
    print(f"saved {plot_path}")


if __name__ == "__main__":
    main()
