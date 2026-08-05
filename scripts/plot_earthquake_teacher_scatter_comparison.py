#!/usr/bin/env python3
"""Create the four-panel Earthquake scatter comparison from saved artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import torch


DEFAULT_PREFIX = "earthquake_linear_beta_100k_ema"
DEFAULT_COMPARISON_DIR = Path(f"results/{DEFAULT_PREFIX}_comparison")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--heat-dir",
        type=Path,
        default=Path(f"results/{DEFAULT_PREFIX}_heat"),
    )
    parser.add_argument(
        "--varadhan-dir",
        type=Path,
        default=Path(f"results/{DEFAULT_PREFIX}_varadhan"),
    )
    parser.add_argument(
        "--malliavin-dir",
        type=Path,
        default=Path(f"results/{DEFAULT_PREFIX}_malliavin"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_COMPARISON_DIR / "scatter_comparison.png",
    )
    parser.add_argument("--max-points", type=int, default=4096)
    parser.add_argument("--marker-size", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--view-lon", type=float, default=70.0)
    parser.add_argument("--view-lat", type=float, default=30.0)
    parser.add_argument("--no-pdf", action="store_true")
    return parser.parse_args()


def _load_tensor(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"missing artifact: {path}")
    value = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected tensor artifact at {path}, got {type(value).__name__}")
    return value


def load_observed_points(run_dirs: Mapping[str, Path]) -> torch.Tensor:
    """Load observed points saved before teacher generation, including --skip-viz runs."""

    for teacher in ("heat", "varadhan", "malliavin"):
        path = run_dirs[teacher] / "teacher_initial_points.pt"
        if not path.is_file():
            continue
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict) or "train_initial_points" not in payload:
            raise TypeError(f"unexpected observed-points artifact format: {path}")
        parts = [payload["train_initial_points"]]
        validation = payload.get("validation_initial_points")
        if validation is not None:
            parts.append(validation)
        return torch.cat(parts, dim=0)
    raise FileNotFoundError(
        "teacher_initial_points.pt was not found in any supplied run directory"
    )


def load_saved_scatter_artifacts(
    run_dirs: Mapping[str, Path],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    observed = load_observed_points(run_dirs)
    generated = {
        teacher: _load_tensor(run_dirs[teacher] / "generated_samples.pt")
        for teacher in ("heat", "varadhan", "malliavin")
    }
    return observed, generated


def main() -> None:
    args = parse_args()
    run_dirs = {
        "heat": args.heat_dir.expanduser().resolve(),
        "varadhan": args.varadhan_dir.expanduser().resolve(),
        "malliavin": args.malliavin_dir.expanduser().resolve(),
    }
    observed, generated = load_saved_scatter_artifacts(run_dirs)

    # Delay visualization imports so artifact-loading tests do not require a
    # working display backend.
    from scoremodel_ext.manifold.earthquake_smoke_viz import (
        generate_earthquake_scatter_comparison,
    )

    result = generate_earthquake_scatter_comparison(
        observed_points=observed,
        generated_by_teacher=generated,
        output_path=args.output.expanduser().resolve(),
        max_points=args.max_points,
        marker_size=args.marker_size,
        alpha=args.alpha,
        view_lon=args.view_lon,
        view_lat=args.view_lat,
        save_pdf=not args.no_pdf,
    )
    print(
        f"saved {result['output_path']} "
        f"({result['displayed_count_per_panel']} points per panel)"
    )
    if result["pdf_path"] is not None:
        print(f"saved {result['pdf_path']}")


if __name__ == "__main__":
    main()
