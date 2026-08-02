#!/usr/bin/env python3
"""Compare two Phase 2B/C runs and export a compact metrics table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _pairwise_to_heat(metrics: dict) -> float:
    pairwise = metrics.get("reverse", {}).get("pairwise_between_methods", {})
    if "heat" in pairwise and "trained_malliavin" in pairwise["heat"]:
        return float(pairwise["heat"]["trained_malliavin"])
    trained = pairwise.get("trained_malliavin", {})
    if "heat" in trained:
        return float(trained["heat"])
    return float("nan")


def _row(method_name: str, metrics: dict) -> Dict[str, object]:
    training = metrics.get("training", {})
    reverse_by_method = metrics.get("reverse", {}).get("by_method", {})
    trained_reverse = reverse_by_method.get("trained_malliavin", {})
    return {
        "method": method_name,
        "heat_score_mse": training.get("heat_score_mse"),
        "heat_score_mean_cosine": training.get("heat_score_mean_cosine"),
        "mean_geodesic_distance_to_initial": trained_reverse.get("mean_geodesic_distance_to_initial"),
        "rmse_geodesic_distance_to_initial": trained_reverse.get("rmse_geodesic_distance_to_initial"),
        "max_geodesic_distance_to_initial": trained_reverse.get("max_geodesic_distance_to_initial"),
        "pairwise_distance_to_heat": _pairwise_to_heat(metrics),
        "training_seconds": metrics.get("timing_seconds", {}).get("training"),
    }


def _markdown(rows: List[Dict[str, object]]) -> str:
    headers = [
        "method",
        "heat_score_mse",
        "heat_score_mean_cosine",
        "mean_geodesic_distance_to_initial",
        "rmse_geodesic_distance_to_initial",
        "max_geodesic_distance_to_initial",
        "pairwise_distance_to_heat",
        "training_seconds",
    ]
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join([head, sep, *body]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a-metrics", type=Path, required=True)
    parser.add_argument("--run-a-name", type=str, default="skorokhod")
    parser.add_argument("--run-b-metrics", type=Path, required=True)
    parser.add_argument("--run-b-name", type=str, default="direct_score")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_a = _load_json(args.run_a_metrics.resolve())
    metrics_b = _load_json(args.run_b_metrics.resolve())

    rows = [
        _row(args.run_a_name, metrics_a),
        _row(args.run_b_name, metrics_b),
    ]

    csv_path = output_dir / "phase2bc_method_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_dir / "phase2bc_method_comparison.md"
    md_path.write_text(_markdown(rows), encoding="utf-8")

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
