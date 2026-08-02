#!/usr/bin/env python3
"""Aggregate Phase 2A seed runs into tables for paper-ready reporting."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Mapping


PRIMARY_KEYS = [
    "malliavin_vs_heat_rmse",
    "malliavin_vs_heat_mean_cosine",
    "malliavin_vs_varadhan_rmse",
    "varadhan_vs_heat_rmse",
    "max_endpoint_norm_error",
    "max_tangent_residual",
    "nan_rate",
    "generation_seconds",
    "total_seconds",
]

BIN_COMPARE_KEYS = [
    "count",
    "malliavin_vs_heat_rmse",
    "malliavin_vs_heat_mean_cosine",
    "malliavin_vs_varadhan_rmse",
    "varadhan_vs_heat_rmse",
]


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _fmt(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return f"{value:.10g}"
    return str(value)


def _markdown_table(headers: List[str], rows: Iterable[Iterable[object]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(_fmt(cell) for cell in row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": mean(values),
        "std": stdev(values) if len(values) >= 2 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _seed_label(run_dir: Path, metrics: Mapping[str, object]) -> str:
    if "seed" in metrics:
        return f"seed{int(metrics['seed'])}"
    return run_dir.name


def _write_csv(path: Path, headers: List[str], rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def summarize_runs(run_dirs: List[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows: List[Dict[str, object]] = []
    bins_long_rows: List[Dict[str, object]] = []

    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"missing metrics.json: {metrics_path}")
        metrics = _load_json(metrics_path)
        seed_label = _seed_label(run_dir, metrics)
        row = {"run_dir": str(run_dir), "seed": seed_label}
        row.update({key: metrics.get(key) for key in PRIMARY_KEYS})
        per_seed_rows.append(row)

        geodesic_bins = list(metrics.get("geodesic_bins", []))
        for index, entry in enumerate(geodesic_bins):
            bins_long_rows.append(
                {
                    "run_dir": str(run_dir),
                    "seed": seed_label,
                    "bin_index": index,
                    "start_degrees": entry.get("start_degrees"),
                    "end_degrees": entry.get("end_degrees"),
                    "start_radians": entry.get("start_radians"),
                    "end_radians": entry.get("end_radians"),
                    **{key: entry.get(key) for key in BIN_COMPARE_KEYS},
                }
            )

    per_seed_csv = output_dir / "phase2a_seed_metrics.csv"
    _write_csv(per_seed_csv, ["run_dir", "seed", *PRIMARY_KEYS], per_seed_rows)

    summary_rows: List[Dict[str, object]] = []
    for key in PRIMARY_KEYS:
        values = [float(row[key]) for row in per_seed_rows]
        stats = _stats(values)
        summary_rows.append({"metric": key, **stats})

    summary_csv = output_dir / "phase2a_metric_summary.csv"
    _write_csv(summary_csv, ["metric", "mean", "std", "min", "max"], summary_rows)

    bins_long_csv = output_dir / "phase2a_geodesic_bins_long.csv"
    _write_csv(
        bins_long_csv,
        [
            "run_dir",
            "seed",
            "bin_index",
            "start_degrees",
            "end_degrees",
            "start_radians",
            "end_radians",
            *BIN_COMPARE_KEYS,
        ],
        bins_long_rows,
    )

    # Aggregate by bin index for cross-seed comparisons.
    grouped: Dict[int, List[Dict[str, object]]] = {}
    for row in bins_long_rows:
        grouped.setdefault(int(row["bin_index"]), []).append(row)

    bins_summary_rows: List[Dict[str, object]] = []
    for bin_index in sorted(grouped):
        group = grouped[bin_index]
        entry: Dict[str, object] = {
            "bin_index": bin_index,
            "start_degrees_mean": mean(float(g["start_degrees"]) for g in group),
            "end_degrees_mean": mean(float(g["end_degrees"]) for g in group),
        }
        for key in BIN_COMPARE_KEYS:
            values = [float(g[key]) for g in group]
            stats = _stats(values)
            entry[f"{key}_mean"] = stats["mean"]
            entry[f"{key}_std"] = stats["std"]
            entry[f"{key}_min"] = stats["min"]
            entry[f"{key}_max"] = stats["max"]
        bins_summary_rows.append(entry)

    bins_summary_csv = output_dir / "phase2a_geodesic_bins_summary.csv"
    bin_headers = [
        "bin_index",
        "start_degrees_mean",
        "end_degrees_mean",
    ]
    for key in BIN_COMPARE_KEYS:
        bin_headers.extend(
            [
                f"{key}_mean",
                f"{key}_std",
                f"{key}_min",
                f"{key}_max",
            ]
        )
    _write_csv(bins_summary_csv, bin_headers, bins_summary_rows)

    # Markdown report with both primary and geodesic-bin summaries.
    per_seed_md = output_dir / "phase2a_seed_report.md"
    per_seed_table = _markdown_table(
        ["seed", *PRIMARY_KEYS],
        [[row["seed"], *[row[key] for key in PRIMARY_KEYS]] for row in per_seed_rows],
    )
    summary_table = _markdown_table(
        ["metric", "mean", "std", "min", "max"],
        [[row["metric"], row["mean"], row["std"], row["min"], row["max"]] for row in summary_rows],
    )
    bins_table = _markdown_table(
        [
            "bin_index",
            "start_deg_mean",
            "end_deg_mean",
            "rmse_mean",
            "rmse_std",
            "cosine_mean",
            "cosine_std",
            "count_mean",
        ],
        [
            [
                row["bin_index"],
                row["start_degrees_mean"],
                row["end_degrees_mean"],
                row["malliavin_vs_heat_rmse_mean"],
                row["malliavin_vs_heat_rmse_std"],
                row["malliavin_vs_heat_mean_cosine_mean"],
                row["malliavin_vs_heat_mean_cosine_std"],
                row["count_mean"],
            ]
            for row in bins_summary_rows
        ],
    )
    per_seed_md.write_text(
        "\n\n".join(
            [
                "# Phase 2A seed comparison",
                "## Per-seed metrics",
                per_seed_table,
                "## Summary statistics (mean/std/min/max)",
                summary_table,
                "## Geodesic-bin cross-seed summary",
                bins_table,
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"wrote {per_seed_csv}")
    print(f"wrote {summary_csv}")
    print(f"wrote {bins_long_csv}")
    print(f"wrote {bins_summary_csv}")
    print(f"wrote {per_seed_md}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        type=Path,
        default=[
            Path("results/s2_malliavin_teacher_baseline"),
            Path("results/s2_malliavin_teacher_seed1"),
            Path("results/s2_malliavin_teacher_seed2"),
        ],
        help="Phase 2A run directories containing metrics.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/s2_malliavin_teacher_phase2a_summary"),
        help="Directory for aggregate CSV/Markdown outputs",
    )
    args = parser.parse_args()

    summarize_runs([path.resolve() for path in args.run_dirs], args.output_dir.resolve())


if __name__ == "__main__":
    main()