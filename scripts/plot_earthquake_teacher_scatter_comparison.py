#!/usr/bin/env python3
"""Create the four-panel Earthquake scatter comparison from saved artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from scoremodel_ext.manifold.earthquake_comparison_artifacts import (
    DEFAULT_COMPARISON_DIR,
    DEFAULT_PREFIX,
    load_saved_scatter_artifacts,
)


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
