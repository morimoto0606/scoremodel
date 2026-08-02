#!/usr/bin/env python3
"""Render the Phase 2A paper figures from a saved S^2 teacher run."""

from __future__ import annotations

import argparse
from pathlib import Path

from scoremodel_ext.manifold.phase2a_viz import generate_phase2a_visuals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/s2_malliavin_teacher_baseline"),
        help="Directory containing teacher_dataset.pt, metrics.json, and run_config.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated figures and tables (defaults to <input-dir>/plots)",
    )
    parser.add_argument("--max-arrow-points", type=int, default=160)
    parser.add_argument("--arrow-length", type=float, default=0.18)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve() if args.output_dir is not None else None
    generate_phase2a_visuals(
        args.input_dir.resolve(),
        output_dir=output_dir,
        max_arrow_points=args.max_arrow_points,
        arrow_length=args.arrow_length,
    )
    resolved_output = output_dir or (args.input_dir.resolve() / "plots")
    print(f"wrote Phase 2A visuals to {resolved_output}")


if __name__ == "__main__":
    main()