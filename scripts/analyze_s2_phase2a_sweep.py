#!/usr/bin/env python3
"""Aggregate and plot Phase 2A sweep results for paper-style reporting.

Supported sweeps:
- n_paths
- covariance_regularization
- n_steps
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, Iterable, List, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


CONDITION_NUMBER_CHOICES = (
    "none",
    "unregularized_tangent",
    "regularized_tangent",
)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _fmt(value: object) -> str:
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.10g}"
        return str(value)
    return str(value)


def _markdown_table(headers: List[str], rows: Iterable[Iterable[object]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(_fmt(cell) for cell in row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def _stats(values: List[float]) -> Dict[str, float]:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": mean(finite_values),
        "std": stdev(finite_values) if len(finite_values) >= 2 else 0.0,
        "min": min(finite_values),
        "max": max(finite_values),
    }


def _write_csv(path: Path, headers: List[str], rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _condition_number_pathwise_from_dataset(
    dataset_path: Path,
    *,
    definition: str,
    covariance_regularization: float,
) -> torch.Tensor:
    dataset = torch.load(dataset_path, map_location="cpu")
    eig = dataset.get("covariance_eigenvalues")
    if eig is None:
        raise ValueError(
            f"teacher_dataset.pt does not contain pathwise covariance_eigenvalues: {dataset_path}"
        )
    eig = torch.as_tensor(eig)
    if eig.ndim != 2 or eig.shape[1] < 3:
        raise ValueError(
            f"invalid covariance_eigenvalues shape {tuple(eig.shape)} in {dataset_path}; expected [n_paths, >=3]"
        )
    # Do not assume storage order; sort ascending per path first.
    eig_sorted, _ = torch.sort(eig[:, :3], dim=1)
    lambda2 = eig_sorted[:, 1].abs()
    lambda3 = eig_sorted[:, 2].abs()
    eps = 1e-15

    if definition == "unregularized_tangent":
        return lambda3.clamp_min(eps) / lambda2.clamp_min(eps)
    if definition == "regularized_tangent":
        reg = float(covariance_regularization)
        return (lambda3 + reg).clamp_min(eps) / (lambda2 + reg).clamp_min(eps)
    raise ValueError(f"unsupported condition-number definition: {definition}")


def _condition_number_summary(pathwise_values: torch.Tensor) -> Dict[str, float]:
    if pathwise_values.numel() == 0:
        return {
            "tangent_pathwise_mean": float("nan"),
            "tangent_pathwise_median": float("nan"),
            "tangent_pathwise_max": float("nan"),
        }
    finite = pathwise_values[torch.isfinite(pathwise_values)]
    if finite.numel() == 0:
        return {
            "tangent_pathwise_mean": float("nan"),
            "tangent_pathwise_median": float("nan"),
            "tangent_pathwise_max": float("nan"),
        }
    return {
        "tangent_pathwise_mean": float(finite.mean().item()),
        "tangent_pathwise_median": float(median(finite.tolist())),
        "tangent_pathwise_max": float(finite.max().item()),
    }


def _resolve_sweep_value(metrics: dict, run_config: dict, key: str) -> float:
    if key in run_config:
        return float(run_config[key])
    if key in metrics:
        return float(metrics[key])
    raise KeyError(f"missing sweep key '{key}' in run_config.json and metrics.json")


def _resolve_covariance_regularization(metrics: Mapping[str, object], run_config: Mapping[str, object]) -> float:
    if "covariance_regularization" in run_config:
        return float(run_config["covariance_regularization"])
    if "covariance_regularization" in metrics:
        return float(metrics["covariance_regularization"])
    raise KeyError("missing covariance_regularization in run_config.json and metrics.json")


def collect_rows(
    run_dirs: List[Path],
    sweep_key: str,
    *,
    condition_number_definition: str,
    on_missing: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.json"
        config_path = run_dir / "run_config.json"
        dataset_path = run_dir / "teacher_dataset.pt"
        if not metrics_path.exists():
            if on_missing == "skip":
                print(f"skip missing metrics.json: {metrics_path}")
                continue
            raise FileNotFoundError(f"missing metrics.json: {metrics_path}")
        if not config_path.exists():
            if on_missing == "skip":
                print(f"skip missing run_config.json: {config_path}")
                continue
            raise FileNotFoundError(f"missing run_config.json: {config_path}")

        metrics = _load_json(metrics_path)
        run_config = _load_json(config_path)
        row = {
            "run_dir": str(run_dir),
            "seed": int(metrics.get("seed", run_config.get("seed", -1))),
            sweep_key: _resolve_sweep_value(metrics, run_config, sweep_key),
            "malliavin_vs_heat_rmse": float(metrics["malliavin_vs_heat_rmse"]),
            "malliavin_vs_heat_mean_cosine": float(metrics["malliavin_vs_heat_mean_cosine"]),
            "malliavin_vs_varadhan_rmse": float(metrics.get("malliavin_vs_varadhan_rmse", float("nan"))),
            "varadhan_vs_heat_rmse": float(metrics.get("varadhan_vs_heat_rmse", float("nan"))),
            "generation_seconds": float(metrics["generation_seconds"]),
            "total_seconds": float(metrics["total_seconds"]),
            "nan_rate": float(metrics.get("nan_rate", float("nan"))),
            "max_endpoint_norm_error": float(metrics.get("max_endpoint_norm_error", float("nan"))),
            "max_tangent_residual": float(metrics.get("max_tangent_residual", float("nan"))),
        }
        if condition_number_definition == "none":
            row["condition_number_definition"] = "none"
            row["s2_tangent_condition_number_pathwise_mean"] = float("nan")
            row["s2_tangent_condition_number_pathwise_median"] = float("nan")
            row["s2_tangent_condition_number_pathwise_max"] = float("nan")
        elif dataset_path.exists():
            reg = _resolve_covariance_regularization(metrics, run_config)
            pathwise = _condition_number_pathwise_from_dataset(
                dataset_path,
                definition=condition_number_definition,
                covariance_regularization=reg,
            )
            summary = _condition_number_summary(pathwise)
            row["condition_number_definition"] = condition_number_definition
            row["s2_tangent_condition_number_pathwise_mean"] = summary["tangent_pathwise_mean"]
            row["s2_tangent_condition_number_pathwise_median"] = summary["tangent_pathwise_median"]
            row["s2_tangent_condition_number_pathwise_max"] = summary["tangent_pathwise_max"]
        else:
            if on_missing == "skip":
                print(f"skip missing teacher_dataset.pt for condition number: {dataset_path}")
                continue
            raise FileNotFoundError(f"missing teacher_dataset.pt: {dataset_path}")
        rows.append(row)

    if not rows:
        raise ValueError("no valid runs were collected")
    return rows


def aggregate_by_sweep(rows: List[Dict[str, object]], sweep_key: str) -> List[Dict[str, object]]:
    grouped: Dict[float, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(float(row[sweep_key]), []).append(row)

    summary: List[Dict[str, object]] = []
    metric_keys = [
        "malliavin_vs_heat_rmse",
        "malliavin_vs_heat_mean_cosine",
        "malliavin_vs_varadhan_rmse",
        "varadhan_vs_heat_rmse",
        "generation_seconds",
        "total_seconds",
        "s2_tangent_condition_number_pathwise_mean",
        "s2_tangent_condition_number_pathwise_median",
        "s2_tangent_condition_number_pathwise_max",
    ]
    for sweep_value in sorted(grouped.keys()):
        group = grouped[sweep_value]
        entry: Dict[str, object] = {sweep_key: sweep_value, "n_runs": len(group)}
        for key in metric_keys:
            values = [float(item[key]) for item in group]
            stats = _stats(values)
            entry[f"{key}_mean"] = stats["mean"]
            entry[f"{key}_std"] = stats["std"]
            entry[f"{key}_min"] = stats["min"]
            entry[f"{key}_max"] = stats["max"]
        summary.append(entry)
    return summary


def _same_geodesic_edges(metrics_a: Sequence[Mapping[str, object]], metrics_b: Sequence[Mapping[str, object]], tol: float = 1e-12) -> bool:
    if len(metrics_a) != len(metrics_b):
        return False
    for left, right in zip(metrics_a, metrics_b):
        for key in ("start_radians", "end_radians"):
            if abs(float(left[key]) - float(right[key])) > tol:
                return False
    return True


def validate_geodesic_bin_compatibility(run_dirs: List[Path]) -> None:
    reference_bins = None
    reference_path = None
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        bins = list(_load_json(metrics_path).get("geodesic_bins", []))
        if not bins:
            continue
        if reference_bins is None:
            reference_bins = bins
            reference_path = metrics_path
            continue
        if not _same_geodesic_edges(reference_bins, bins):
            print(
                "warning: geodesic bin edges differ across runs; "
                "this script does not aggregate bin-level metrics across incompatible edges."
            )
            print(f"reference: {reference_path}")
            print(f"mismatch : {metrics_path}")
            return


def plot_n_paths(summary: List[Dict[str, object]], sweep_key: str, out_png: Path) -> None:
    xs = [float(row[sweep_key]) for row in summary]
    rmse = [float(row["malliavin_vs_heat_rmse_mean"]) for row in summary]
    cosine = [float(row["malliavin_vs_heat_mean_cosine_mean"]) for row in summary]
    runtime = [float(row["total_seconds_mean"]) for row in summary]

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(xs, rmse, marker="o")
    axes[0].set_title("RMSE vs n_paths")
    axes[0].set_xlabel("n_paths")
    axes[0].set_ylabel("malliavin_vs_heat_rmse")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(xs, cosine, marker="o")
    axes[1].set_title("Cosine vs n_paths")
    axes[1].set_xlabel("n_paths")
    axes[1].set_ylabel("malliavin_vs_heat_mean_cosine")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(xs, runtime, marker="o")
    axes[2].set_title("Runtime vs n_paths")
    axes[2].set_xlabel("n_paths")
    axes[2].set_ylabel("total_seconds")
    axes[2].grid(True, alpha=0.25)

    figure.tight_layout()
    figure.savefig(out_png, dpi=200)
    plt.close(figure)


def _filter_positive_log_x(summary: List[Dict[str, object]], sweep_key: str) -> List[Dict[str, object]]:
    filtered = [row for row in summary if float(row[sweep_key]) > 0.0 and math.isfinite(float(row[sweep_key]))]
    if len(filtered) != len(summary):
        print("warning: excluded non-positive or non-finite x values for log-scale plot")
    if not filtered:
        raise ValueError("no positive finite sweep values available for log-scale plotting")
    return filtered


def plot_covariance_regularization(summary: List[Dict[str, object]], sweep_key: str, out_png: Path) -> None:
    summary = _filter_positive_log_x(summary, sweep_key)
    xs = [float(row[sweep_key]) for row in summary]
    rmse = [float(row["malliavin_vs_heat_rmse_mean"]) for row in summary]
    cosine = [float(row["malliavin_vs_heat_mean_cosine_mean"]) for row in summary]
    cond = [float(row["s2_tangent_condition_number_pathwise_mean_mean"]) for row in summary]
    gen_time = [float(row["generation_seconds_mean"]) for row in summary]

    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    axes = axes.reshape(-1)
    for axis in axes:
        axis.set_xscale("log")

    axes[0].plot(xs, rmse, marker="o")
    axes[0].set_title("RMSE vs covariance_regularization")
    axes[0].set_xlabel("covariance_regularization")
    axes[0].set_ylabel("malliavin_vs_heat_rmse")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(xs, cosine, marker="o")
    axes[1].set_title("Cosine vs covariance_regularization")
    axes[1].set_xlabel("covariance_regularization")
    axes[1].set_ylabel("malliavin_vs_heat_mean_cosine")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(xs, cond, marker="o")
    axes[2].set_title("Condition number trend")
    axes[2].set_xlabel("covariance_regularization")
    axes[2].set_ylabel("mean tangent condition number")
    axes[2].grid(True, alpha=0.25)

    axes[3].plot(xs, gen_time, marker="o")
    axes[3].set_title("Generation time vs covariance_regularization")
    axes[3].set_xlabel("covariance_regularization")
    axes[3].set_ylabel("generation_seconds")
    axes[3].grid(True, alpha=0.25)

    figure.tight_layout()
    figure.savefig(out_png, dpi=200)
    plt.close(figure)


def plot_n_steps(summary: List[Dict[str, object]], sweep_key: str, out_png: Path) -> None:
    xs = [float(row[sweep_key]) for row in summary]
    rmse = [float(row["malliavin_vs_heat_rmse_mean"]) for row in summary]
    cosine = [float(row["malliavin_vs_heat_mean_cosine_mean"]) for row in summary]
    runtime = [float(row["total_seconds_mean"]) for row in summary]

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(xs, rmse, marker="o")
    axes[0].set_title("RMSE vs n_steps")
    axes[0].set_xlabel("n_steps")
    axes[0].set_ylabel("malliavin_vs_heat_rmse")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(xs, cosine, marker="o")
    axes[1].set_title("Cosine vs n_steps")
    axes[1].set_xlabel("n_steps")
    axes[1].set_ylabel("malliavin_vs_heat_mean_cosine")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(xs, runtime, marker="o")
    axes[2].set_title("Runtime vs n_steps")
    axes[2].set_xlabel("n_steps")
    axes[2].set_ylabel("total_seconds")
    axes[2].grid(True, alpha=0.25)

    figure.tight_layout()
    figure.savefig(out_png, dpi=200)
    plt.close(figure)


def write_outputs(
    output_dir: Path,
    sweep_key: str,
    rows: List[Dict[str, object]],
    summary: List[Dict[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = output_dir / f"{sweep_key}_raw.csv"
    raw_headers = [
        "run_dir",
        "seed",
        sweep_key,
        "malliavin_vs_heat_rmse",
        "malliavin_vs_heat_mean_cosine",
        "malliavin_vs_varadhan_rmse",
        "varadhan_vs_heat_rmse",
        "nan_rate",
        "max_endpoint_norm_error",
        "max_tangent_residual",
        "condition_number_definition",
        "s2_tangent_condition_number_pathwise_mean",
        "s2_tangent_condition_number_pathwise_median",
        "s2_tangent_condition_number_pathwise_max",
        "generation_seconds",
        "total_seconds",
    ]
    _write_csv(raw_csv, raw_headers, rows)

    summary_csv = output_dir / f"{sweep_key}_summary.csv"
    summary_headers = [sweep_key, "n_runs"]
    for key in [
        "malliavin_vs_heat_rmse",
        "malliavin_vs_heat_mean_cosine",
        "malliavin_vs_varadhan_rmse",
        "varadhan_vs_heat_rmse",
        "s2_tangent_condition_number_pathwise_mean",
        "s2_tangent_condition_number_pathwise_median",
        "s2_tangent_condition_number_pathwise_max",
        "generation_seconds",
        "total_seconds",
    ]:
        summary_headers.extend([f"{key}_mean", f"{key}_std", f"{key}_min", f"{key}_max"])
    _write_csv(summary_csv, summary_headers, summary)

    report_md = output_dir / f"{sweep_key}_report.md"
    raw_table = _markdown_table(
        [
            sweep_key,
            "seed",
            "rmse",
            "cosine",
            "cond_mean",
            "cond_median",
            "cond_max",
            "gen_s",
            "total_s",
        ],
        [
            [
                row[sweep_key],
                row["seed"],
                row["malliavin_vs_heat_rmse"],
                row["malliavin_vs_heat_mean_cosine"],
                row["s2_tangent_condition_number_pathwise_mean"],
                row["s2_tangent_condition_number_pathwise_median"],
                row["s2_tangent_condition_number_pathwise_max"],
                row["generation_seconds"],
                row["total_seconds"],
            ]
            for row in sorted(rows, key=lambda r: (float(r[sweep_key]), int(r["seed"])))
        ],
    )
    summary_table = _markdown_table(
        [
            sweep_key,
            "n_runs",
            "rmse_mean",
            "rmse_std",
            "cosine_mean",
            "cosine_std",
            "cond_mean",
            "cond_median",
            "cond_max",
            "gen_s_mean",
            "total_s_mean",
        ],
        [
            [
                row[sweep_key],
                row["n_runs"],
                row["malliavin_vs_heat_rmse_mean"],
                row["malliavin_vs_heat_rmse_std"],
                row["malliavin_vs_heat_mean_cosine_mean"],
                row["malliavin_vs_heat_mean_cosine_std"],
                row["s2_tangent_condition_number_pathwise_mean_mean"],
                row["s2_tangent_condition_number_pathwise_median_mean"],
                row["s2_tangent_condition_number_pathwise_max_mean"],
                row["generation_seconds_mean"],
                row["total_seconds_mean"],
            ]
            for row in summary
        ],
    )
    report_md.write_text(
        "\n\n".join(
            [
                f"# Phase 2A sweep: {sweep_key}",
                "Condition number columns use S2 tangent-restricted definitions.",
                "- unregularized_tangent: kappa_tan = lambda3 / max(lambda2, eps), with per-path eigenvalue sort lambda1 <= lambda2 <= lambda3",
                "- regularized_tangent: kappa_tan_lambda = (lambda3 + lambda) / (lambda2 + lambda)",
                "## Per-run metrics",
                raw_table,
                "## Grouped summary (mean/std/min/max)",
                summary_table,
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    plot_path = output_dir / f"{sweep_key}_plots.png"
    if sweep_key == "n_paths":
        plot_n_paths(summary, sweep_key, plot_path)
    elif sweep_key == "covariance_regularization":
        plot_covariance_regularization(summary, sweep_key, plot_path)
    elif sweep_key == "n_steps":
        plot_n_steps(summary, sweep_key, plot_path)
    else:
        raise ValueError(f"unsupported sweep key: {sweep_key}")

    print(f"wrote {raw_csv}")
    print(f"wrote {summary_csv}")
    print(f"wrote {report_md}")
    print(f"wrote {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-key",
        choices=("n_paths", "covariance_regularization", "n_steps"),
        required=True,
    )
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        type=Path,
        required=True,
        help="Run directories that each contain teacher_dataset.pt, metrics.json, run_config.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--condition-number-definition",
        choices=CONDITION_NUMBER_CHOICES,
        default="none",
        help=(
            "Condition-number definition for covariance plots: "
            "none | unregularized_tangent | regularized_tangent"
        ),
    )
    parser.add_argument(
        "--on-missing",
        choices=("error", "skip"),
        default="error",
        help="How to handle missing artifacts for a requested run directory.",
    )
    args = parser.parse_args()

    run_dirs = [path.resolve() for path in args.run_dirs]
    validate_geodesic_bin_compatibility(run_dirs)
    rows = collect_rows(
        run_dirs,
        args.sweep_key,
        condition_number_definition=args.condition_number_definition,
        on_missing=args.on_missing,
    )
    summary = aggregate_by_sweep(rows, args.sweep_key)
    write_outputs(args.output_dir.resolve(), args.sweep_key, rows, summary)


if __name__ == "__main__":
    main()