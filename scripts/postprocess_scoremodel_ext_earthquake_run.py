#!/usr/bin/env python3
"""Create standard Earthquake plots for one saved scoremodel_ext run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scoremodel_ext.manifold.earthquake_comparison_artifacts import (
    load_tensor_artifact,
)


DEFAULT_RUN_DIR = Path("results/earthquake_heat_upstream_style_training")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="output directory; defaults to <run-dir>/viz",
    )
    parser.add_argument("--max-scatter-points", type=int, default=4096)
    parser.add_argument("--marker-size", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--grid-size", type=int, default=400)
    parser.add_argument("--kappa", type=float, default=80.0)
    parser.add_argument("--density-bandwidth-scale", type=float, default=0.5)
    parser.add_argument("--view-lon", type=float, default=70.0)
    parser.add_argument("--view-lat", type=float, default=30.0)
    parser.add_argument("--generated-title", type=str, default=None)
    parser.add_argument("--no-pdf", action="store_true")
    return parser.parse_args()


def _validate_s2_points(points: torch.Tensor, *, artifact_name: str) -> None:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"{artifact_name} must have shape (n, 3), got {tuple(points.shape)}"
        )
    if points.shape[0] < 1:
        raise ValueError(f"{artifact_name} must not be empty")
    if not bool(torch.isfinite(points).all()):
        raise ValueError(f"{artifact_name} contains non-finite values")


def load_scoremodel_ext_run_artifacts(
    run_dir: Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load observed and generated standard-Earth xyz without conversion."""

    resolved_run_dir = run_dir.expanduser().resolve()
    generated = load_tensor_artifact(resolved_run_dir / "generated_samples.pt")

    initial_path = resolved_run_dir / "teacher_initial_points.pt"
    if initial_path.is_file():
        initial_payload = torch.load(initial_path, map_location="cpu")
        if not isinstance(initial_payload, dict):
            raise TypeError(f"expected dictionary artifact at {initial_path}")
        train = initial_payload.get("train_initial_points")
        validation = initial_payload.get("validation_initial_points")
        if not isinstance(train, torch.Tensor):
            raise KeyError(f"{initial_path} has no tensor train_initial_points")
        observed_parts = [train]
        if validation is not None:
            if not isinstance(validation, torch.Tensor):
                raise TypeError(
                    f"validation_initial_points in {initial_path} is not a tensor"
                )
            observed_parts.append(validation)
    else:
        # Older/partially copied runs may omit teacher_initial_points.pt while
        # retaining the same points inside the saved teacher datasets.
        observed_parts = []
        for dataset_name in ("teacher_dataset.pt", "validation_dataset.pt"):
            dataset_path = resolved_run_dir / dataset_name
            if not dataset_path.is_file():
                continue
            payload = torch.load(dataset_path, map_location="cpu")
            if not isinstance(payload, dict) or not isinstance(
                payload.get("initial_point"), torch.Tensor
            ):
                raise TypeError(
                    f"expected initial_point tensor in dictionary artifact {dataset_path}"
                )
            observed_parts.append(payload["initial_point"])
        if len(observed_parts) == 1:
            target_path = resolved_run_dir / "target_samples.pt"
            if target_path.is_file():
                observed_parts.append(load_tensor_artifact(target_path))
        if not observed_parts:
            raise FileNotFoundError(
                "missing scoremodel_ext observed points; expected "
                f"{initial_path} or teacher_dataset.pt. This script never "
                "substitutes Upstream coordinates"
            )
    observed = torch.cat(observed_parts, dim=0)

    _validate_s2_points(observed, artifact_name="observed points")
    _validate_s2_points(generated, artifact_name="generated samples")
    return observed, generated


def infer_generated_title(run_dir: Path) -> str:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        return "Generated"
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    if not isinstance(metrics, dict):
        return "Generated"
    teacher = str(metrics.get("teacher", "Generated")).title()
    if (
        metrics.get("score_parameterization") == "upstream_scaled_score"
        or metrics.get("training_path") == "upstream_scaled_score"
    ):
        return f"{teacher} (Upstream-style)"
    return teacher


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = (
        run_dir / "viz"
        if args.output_dir is None
        else args.output_dir.expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    observed, generated = load_scoremodel_ext_run_artifacts(run_dir)
    generated_title = (
        infer_generated_title(run_dir)
        if args.generated_title is None
        else args.generated_title
    )
    generated_by_method = {"generated": generated}
    panel_order = ("observed", "generated")
    panel_titles = {"observed": "Observed", "generated": generated_title}

    from scoremodel_ext.manifold.earthquake_smoke_viz import (
        generate_earthquake_density_bandwidth_outputs,
        generate_earthquake_scatter_comparison,
    )

    scatter_global = generate_earthquake_scatter_comparison(
        observed_points=observed,
        generated_by_teacher=generated_by_method,
        output_path=output_dir / "scatter_global.png",
        max_points=args.max_scatter_points,
        marker_size=args.marker_size,
        alpha=args.alpha,
        view_lon=args.view_lon,
        view_lat=args.view_lat,
        save_pdf=not args.no_pdf,
        panel_order=panel_order,
        panel_titles=panel_titles,
    )
    scatter_japan = generate_earthquake_scatter_comparison(
        observed_points=observed,
        generated_by_teacher=generated_by_method,
        output_path=output_dir / "scatter_japan_zoom.png",
        max_points=args.max_scatter_points,
        marker_size=args.marker_size,
        alpha=args.alpha,
        geographic_extent=(120.0, 150.0, 20.0, 50.0),
        save_pdf=not args.no_pdf,
        panel_order=panel_order,
        panel_titles=panel_titles,
    )
    density = generate_earthquake_density_bandwidth_outputs(
        observed_points=observed,
        generated_by_teacher=generated_by_method,
        global_output_path=output_dir / "density_global.png",
        bandwidth_comparison_path=output_dir / "density_bandwidth_comparison.png",
        grid_size=args.grid_size,
        base_kappa=args.kappa,
        bandwidth_scale=args.density_bandwidth_scale,
        view_lon=args.view_lon,
        view_lat=args.view_lat,
        save_pdf=not args.no_pdf,
        panel_order=panel_order,
        panel_titles=panel_titles,
    )

    for result in (scatter_global, scatter_japan):
        print(f"saved {result['output_path']}")
        if result["pdf_path"] is not None:
            print(f"saved {result['pdf_path']}")
    print(f"saved {density['global_output_path']}")
    if density["global_pdf_path"] is not None:
        print(f"saved {density['global_pdf_path']}")
    print(f"saved {density['bandwidth_comparison_path']}")
    if density["bandwidth_pdf_path"] is not None:
        print(f"saved {density['bandwidth_pdf_path']}")


if __name__ == "__main__":
    main()
