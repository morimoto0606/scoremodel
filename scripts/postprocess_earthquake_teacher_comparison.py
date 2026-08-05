#!/usr/bin/env python3
"""Aggregate Earthquake 3-teacher comparison plots and metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from scripts.plot_earthquake_teacher_scatter_comparison import (
    DEFAULT_COMPARISON_DIR,
    DEFAULT_PREFIX,
    load_saved_scatter_artifacts,
)


TEACHERS = ("heat", "varadhan", "malliavin")
METRIC_KEYS = (
    "train_loss",
    "validation_loss",
    "s2_rbf_mmd",
    "nearest_neighbor_geodesic_mean",
    "nearest_neighbor_geodesic_median",
    "reverse_sampling_seconds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for teacher in TEACHERS:
        parser.add_argument(
            f"--{teacher}-dir",
            type=Path,
            default=Path(f"results/{DEFAULT_PREFIX}_{teacher}"),
        )
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=DEFAULT_COMPARISON_DIR,
    )
    parser.add_argument("--max-scatter-points", type=int, default=4096)
    parser.add_argument("--marker-size", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--grid-size", type=int, default=400)
    parser.add_argument("--kappa", type=float, default=80.0)
    parser.add_argument("--view-lon", type=float, default=70.0)
    parser.add_argument("--view-lat", type=float, default=30.0)
    parser.add_argument("--no-pdf", action="store_true")
    return parser.parse_args()


def build_metrics_comparison(run_dirs: Mapping[str, Path]) -> dict:
    """Read and normalize the requested metrics from each teacher run."""

    comparison = {}
    for teacher in TEACHERS:
        metrics_path = run_dirs[teacher] / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"missing metrics artifact: {metrics_path}")
        with metrics_path.open("r", encoding="utf-8") as handle:
            source = json.load(handle)

        train_loss = source.get("train_loss", source.get("final_train_loss"))
        values = {"train_loss": train_loss}
        values.update({key: source.get(key) for key in METRIC_KEYS[1:]})
        missing = [key for key, value in values.items() if value is None]
        if missing:
            raise KeyError(f"{metrics_path} is missing comparison metrics: {missing}")
        comparison[teacher] = values
    return comparison


def save_metrics_comparison(run_dirs: Mapping[str, Path], output_path: Path) -> dict:
    comparison = build_metrics_comparison(run_dirs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2)
    return comparison


def main() -> None:
    args = parse_args()
    run_dirs = {
        teacher: getattr(args, f"{teacher}_dir").expanduser().resolve()
        for teacher in TEACHERS
    }
    comparison_dir = args.comparison_dir.expanduser().resolve()
    comparison_dir.mkdir(parents=True, exist_ok=True)
    observed, generated = load_saved_scatter_artifacts(run_dirs)

    # Plotting imports are delayed so metrics/artifact tests remain headless.
    from scoremodel_ext.manifold.earthquake_smoke_viz import (
        generate_earthquake_density_comparison,
        generate_earthquake_scatter_comparison,
    )

    scatter = generate_earthquake_scatter_comparison(
        observed_points=observed,
        generated_by_teacher=generated,
        output_path=comparison_dir / "scatter_comparison.png",
        max_points=args.max_scatter_points,
        marker_size=args.marker_size,
        alpha=args.alpha,
        view_lon=args.view_lon,
        view_lat=args.view_lat,
        save_pdf=not args.no_pdf,
    )
    density = generate_earthquake_density_comparison(
        observed_points=observed,
        generated_by_teacher=generated,
        output_path=comparison_dir / "density_comparison.png",
        grid_size=args.grid_size,
        kappa=args.kappa,
        view_lon=args.view_lon,
        view_lat=args.view_lat,
        save_pdf=not args.no_pdf,
    )
    metrics_path = comparison_dir / "metrics_comparison.json"
    save_metrics_comparison(run_dirs, metrics_path)

    print(f"saved {scatter['output_path']}")
    if scatter["pdf_path"] is not None:
        print(f"saved {scatter['pdf_path']}")
    print(f"saved {density['output_path']}")
    if density["pdf_path"] is not None:
        print(f"saved {density['pdf_path']}")
    print(f"saved {metrics_path}")


if __name__ == "__main__":
    main()
