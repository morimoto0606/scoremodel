#!/usr/bin/env python3
"""Aggregate Earthquake smoke metrics for heat/varadhan/malliavin runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRIC_COLUMNS = [
    "teacher",
    "final_train_loss",
    "validation_loss",
    "teacher_generation_seconds",
    "training_seconds",
    "reverse_sampling_seconds",
    "s2_rbf_mmd",
    "nearest_neighbor_geodesic_mean",
    "nearest_neighbor_geodesic_median",
    "nearest_neighbor_geodesic_max",
    "generated_sample_norm_error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/earthquake_smoke_comparison"),
        help="Directory containing teacher subdirs: heat, varadhan, malliavin",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/earthquake_smoke_comparison"),
    )
    return parser.parse_args()


def read_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(rows: list[dict], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def write_markdown(rows: list[dict], output_path: Path) -> None:
    header = "| " + " | ".join(METRIC_COLUMNS) + " |\n"
    separator = "|" + "|".join(["---"] * len(METRIC_COLUMNS)) + "|\n"
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Earthquake Smoke Method Comparison\n\n")
        handle.write(header)
        handle.write(separator)
        for row in rows:
            handle.write("| " + " | ".join(_fmt(row[col]) for col in METRIC_COLUMNS) + " |\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for teacher in ("heat", "varadhan", "malliavin"):
        metrics_path = args.root / teacher / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"missing metrics file: {metrics_path}")
        payload = read_metrics(metrics_path)
        row = {"teacher": teacher}
        for column in METRIC_COLUMNS:
            if column == "teacher":
                continue
            row[column] = payload[column]
        rows.append(row)

    csv_path = args.output_dir / "earthquake_method_comparison.csv"
    md_path = args.output_dir / "earthquake_method_comparison.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)

    print(f"saved {csv_path}")
    print(f"saved {md_path}")


if __name__ == "__main__":
    main()