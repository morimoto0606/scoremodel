#!/usr/bin/env python3
"""Postprocess one upstream Earthquake Heat/Varadhan DSM run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

import torch

from scoremodel_ext.manifold.earthquake_adapter import (
    load_earthquake_points,
    nearest_neighbor_geodesic_summary,
    s2_rbf_mmd,
)
from scoremodel_ext.manifold.earthquake_comparison_artifacts import (
    UPSTREAM_ANTIPODAL_COORDINATES,
    load_upstream_generated_samples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--teacher", choices=("heat", "varadhan"), required=True)
    parser.add_argument("--samples-path", type=Path, default=None)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("upstream/riemannian-score-sde/data/quakes_all.csv"),
    )
    parser.add_argument("--max-scatter-points", type=int, default=4096)
    parser.add_argument("--marker-size", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--grid-size", type=int, default=400)
    parser.add_argument("--kappa", type=float, default=80.0)
    parser.add_argument("--density-bandwidth-scale", type=float, default=0.5)
    parser.add_argument("--evaluation-seed", type=int, default=0)
    parser.add_argument("--no-pdf", action="store_true")
    return parser.parse_args()


def _last_train_loss(run_dir: Path) -> float | None:
    candidates = sorted((run_dir / "logs").glob("**/metrics.csv"))
    values = []
    for path in candidates:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                value = row.get("train/loss")
                if value not in (None, ""):
                    values.append(float(value))
    return values[-1] if values else None


def _sample_metadata(samples_path: Path) -> dict:
    metadata_path = samples_path.with_suffix(".json")
    if not metadata_path.is_file():
        return {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {metadata_path}")
    return value


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    samples_path = (
        run_dir / "generated_samples.npy"
        if args.samples_path is None
        else args.samples_path.expanduser().resolve()
    )
    samples = load_upstream_generated_samples(
        samples_path,
        coordinate_system=UPSTREAM_ANTIPODAL_COORDINATES,
    ).to(dtype=torch.float64)
    observed = load_earthquake_points(
        args.data_path.expanduser().resolve(),
        dtype=torch.float64,
        device="cpu",
    )
    panel_order = ("observed", args.teacher)
    panel_titles = {
        "observed": "Observed",
        args.teacher: f"Upstream {args.teacher.title()}",
    }
    generated = {args.teacher: samples}

    from scoremodel_ext.manifold.earthquake_smoke_viz import (
        generate_earthquake_density_bandwidth_outputs,
        generate_earthquake_scatter_comparison,
    )

    generate_earthquake_scatter_comparison(
        observed_points=observed,
        generated_by_teacher=generated,
        output_path=run_dir / "scatter_global.png",
        max_points=args.max_scatter_points,
        marker_size=args.marker_size,
        alpha=args.alpha,
        view_lon=70.0,
        view_lat=30.0,
        save_pdf=not args.no_pdf,
        panel_order=panel_order,
        panel_titles=panel_titles,
    )
    generate_earthquake_scatter_comparison(
        observed_points=observed,
        generated_by_teacher=generated,
        output_path=run_dir / "scatter_japan_zoom.png",
        max_points=args.max_scatter_points,
        marker_size=args.marker_size,
        alpha=args.alpha,
        geographic_extent=(120.0, 150.0, 20.0, 50.0),
        save_pdf=not args.no_pdf,
        panel_order=panel_order,
        panel_titles=panel_titles,
    )
    generate_earthquake_density_bandwidth_outputs(
        observed_points=observed,
        generated_by_teacher=generated,
        global_output_path=run_dir / "density_global.png",
        bandwidth_comparison_path=run_dir / "density_bandwidth_comparison.png",
        grid_size=args.grid_size,
        base_kappa=args.kappa,
        bandwidth_scale=args.density_bandwidth_scale,
        view_lon=70.0,
        view_lat=30.0,
        save_pdf=not args.no_pdf,
        panel_order=panel_order,
        panel_titles=panel_titles,
    )

    geodesic = nearest_neighbor_geodesic_summary(
        samples,
        observed,
        seed=args.evaluation_seed,
    )
    metadata = _sample_metadata(samples_path)
    metrics = {
        "source": "upstream-riemannian-score-sde",
        "teacher": args.teacher,
        "train_loss": _last_train_loss(run_dir),
        "validation_loss": None,
        "s2_rbf_mmd": s2_rbf_mmd(
            samples,
            observed,
            seed=args.evaluation_seed,
        ),
        "nearest_neighbor_geodesic_mean": geodesic["mean"],
        "nearest_neighbor_geodesic_median": geodesic["median"],
        "nearest_neighbor_geodesic_max": geodesic["max"],
        "reverse_sampling_seconds": metadata.get("reverse_sampling_seconds"),
        "sample_count": int(samples.shape[0]),
        "reference_count": int(observed.shape[0]),
        "coordinate_conversion": "upstream-earthquake-antipodal -> standard-earth",
        "reverse_steps": metadata.get("reverse_steps", 100),
        "epsilon": metadata.get("epsilon", 0.001),
        "dtype": metadata.get("dtype"),
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    torch.save(samples, run_dir / "generated_samples.pt")
    torch.save(observed, run_dir / "observed_samples.pt")

    hydra_config = run_dir / ".hydra" / "config.yaml"
    if hydra_config.is_file():
        shutil.copyfile(hydra_config, run_dir / "experiment_config.yaml")

    print(f"saved upstream {args.teacher} artifacts in {run_dir}")


if __name__ == "__main__":
    main()
