#!/usr/bin/env python3
"""Compare upstream Heat/Varadhan with scoremodel_ext Earthquake runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scoremodel_ext.manifold.earthquake_comparison_artifacts import (
    load_tensor_artifact,
)
from scoremodel_ext.manifold.earthquake_adapter import (
    load_earthquake_points,
    nearest_neighbor_geodesic_summary,
    s2_rbf_mmd,
)


METHODS = (
    "upstream_heat",
    "upstream_varadhan",
    "ext_heat",
    "ext_varadhan",
    "ext_malliavin",
)
TITLES = {
    "observed": "Observed",
    "upstream_heat": "Upstream Heat",
    "upstream_varadhan": "Upstream Varadhan",
    "ext_heat": "scoremodel_ext Heat",
    "ext_varadhan": "scoremodel_ext Varadhan",
    "ext_malliavin": "scoremodel_ext Malliavin",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-heat-dir", type=Path, required=True)
    parser.add_argument("--upstream-varadhan-dir", type=Path, required=True)
    parser.add_argument(
        "--ext-heat-dir",
        type=Path,
        default=Path("results/earthquake_linear_beta_100k_ema_heat"),
    )
    parser.add_argument(
        "--ext-varadhan-dir",
        type=Path,
        default=Path("results/earthquake_linear_beta_100k_ema_varadhan"),
    )
    parser.add_argument(
        "--ext-malliavin-dir",
        type=Path,
        default=Path("results/earthquake_linear_beta_100k_ema_malliavin"),
    )
    parser.add_argument("--ext-heat-samples", type=Path, default=None)
    parser.add_argument("--ext-varadhan-samples", type=Path, default=None)
    parser.add_argument("--ext-malliavin-samples", type=Path, default=None)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("upstream/riemannian-score-sde/data/quakes_all.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/earthquake_teacher_comparison"),
    )
    parser.add_argument("--max-scatter-points", type=int, default=4096)
    parser.add_argument("--marker-size", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--grid-size", type=int, default=400)
    parser.add_argument("--kappa", type=float, default=80.0)
    parser.add_argument("--density-bandwidth-scale", type=float, default=0.5)
    parser.add_argument("--no-pdf", action="store_true")
    return parser.parse_args()


def _load_metrics(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing metrics artifact: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected metrics JSON object: {path}")
    return value


def main() -> None:
    args = parse_args()
    run_dirs = {
        "upstream_heat": args.upstream_heat_dir.expanduser().resolve(),
        "upstream_varadhan": args.upstream_varadhan_dir.expanduser().resolve(),
        "ext_heat": args.ext_heat_dir.expanduser().resolve(),
        "ext_varadhan": args.ext_varadhan_dir.expanduser().resolve(),
        "ext_malliavin": args.ext_malliavin_dir.expanduser().resolve(),
    }
    observed = load_earthquake_points(
        args.data_path.expanduser().resolve(),
        dtype=torch.float64,
        device="cpu",
    )
    sample_paths = {
        "upstream_heat": run_dirs["upstream_heat"] / "generated_samples.pt",
        "upstream_varadhan": run_dirs["upstream_varadhan"] / "generated_samples.pt",
        "ext_heat": (
            run_dirs["ext_heat"] / "generated_samples.pt"
            if args.ext_heat_samples is None
            else args.ext_heat_samples.expanduser().resolve()
        ),
        "ext_varadhan": (
            run_dirs["ext_varadhan"] / "generated_samples.pt"
            if args.ext_varadhan_samples is None
            else args.ext_varadhan_samples.expanduser().resolve()
        ),
        "ext_malliavin": (
            run_dirs["ext_malliavin"] / "generated_samples.pt"
            if args.ext_malliavin_samples is None
            else args.ext_malliavin_samples.expanduser().resolve()
        ),
    }
    generated = {
        method: load_tensor_artifact(sample_paths[method]).to(dtype=torch.float64)
        for method in METHODS
    }
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_order = ("observed", *METHODS)

    from scoremodel_ext.manifold.earthquake_smoke_viz import (
        generate_earthquake_density_bandwidth_outputs,
        generate_earthquake_scatter_comparison,
    )

    generate_earthquake_scatter_comparison(
        observed_points=observed,
        generated_by_teacher=generated,
        output_path=output_dir / "scatter_global.png",
        max_points=args.max_scatter_points,
        marker_size=args.marker_size,
        alpha=args.alpha,
        view_lon=70.0,
        view_lat=30.0,
        save_pdf=not args.no_pdf,
        panel_order=panel_order,
        panel_titles=TITLES,
    )
    generate_earthquake_scatter_comparison(
        observed_points=observed,
        generated_by_teacher=generated,
        output_path=output_dir / "scatter_japan_zoom.png",
        max_points=args.max_scatter_points,
        marker_size=args.marker_size,
        alpha=args.alpha,
        geographic_extent=(120.0, 150.0, 20.0, 50.0),
        save_pdf=not args.no_pdf,
        panel_order=panel_order,
        panel_titles=TITLES,
    )
    generate_earthquake_density_bandwidth_outputs(
        observed_points=observed,
        generated_by_teacher=generated,
        global_output_path=output_dir / "density_global.png",
        bandwidth_comparison_path=output_dir / "density_bandwidth_comparison.png",
        grid_size=args.grid_size,
        base_kappa=args.kappa,
        bandwidth_scale=args.density_bandwidth_scale,
        view_lon=70.0,
        view_lat=30.0,
        save_pdf=not args.no_pdf,
        panel_order=panel_order,
        panel_titles=TITLES,
    )

    metrics = {}
    for method in METHODS:
        source_metrics = _load_metrics(run_dirs[method] / "metrics.json")
        geodesic = nearest_neighbor_geodesic_summary(
            generated[method], observed, seed=0
        )
        metrics[method] = {
            **source_metrics,
            "s2_rbf_mmd": s2_rbf_mmd(generated[method], observed, seed=0),
            "nearest_neighbor_geodesic_mean": geodesic["mean"],
            "nearest_neighbor_geodesic_median": geodesic["median"],
            "nearest_neighbor_geodesic_max": geodesic["max"],
            "evaluated_samples_path": str(sample_paths[method]),
            "evaluated_sample_count": int(generated[method].shape[0]),
        }
    with (output_dir / "metrics_comparison.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metrics, handle, indent=2)
    manifest = {
        "comparison_order": list(METHODS),
        "run_directories": {key: str(value) for key, value in run_dirs.items()},
        "sample_artifacts": {key: str(value) for key, value in sample_paths.items()},
        "shared_conditions": {
            "dataset": "Earthquake",
            "manifold": "S2",
            "beta_schedule": "linear(0.001, 5.0, t0=0, tf=1)",
            "reverse_process": "reverse SDE",
            "predictor": "GRW",
            "corrector": None,
            "reverse_steps": 100,
            "epsilon": 0.001,
            "display_sample_count": args.max_scatter_points,
            "evaluated_sample_counts": {
                method: int(generated[method].shape[0]) for method in METHODS
            },
        },
    }
    with (output_dir / "artifact_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)
    for method in METHODS:
        method_dir = output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        with (method_dir / "artifact_manifest.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(
                {
                    "method": method,
                    "run_directory": str(run_dirs[method]),
                    "samples": str(sample_paths[method]),
                    "metrics": str(run_dirs[method] / "metrics.json"),
                },
                handle,
                indent=2,
            )
    print(f"saved comparison artifacts in {output_dir}")


if __name__ == "__main__":
    main()
