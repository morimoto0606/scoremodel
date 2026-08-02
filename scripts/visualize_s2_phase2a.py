#!/usr/bin/env python3
"""Generate Phase 2A sphere visualizations from a saved teacher dataset.

This script reuses the existing numerical backend from
scoremodel_ext.manifold.experiment_s2_malliavin_teacher without changing it.
It reads the saved teacher dataset and metrics/run config from a Phase 2A output
folder and writes quick visual diagnostics to an output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scoremodel_ext.manifold.experiment_s2_malliavin_teacher import (
    save_target_vs_generated_plot,
)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
        help="Directory for generated plots (defaults to <input-dir>/plots)",
    )
    parser.add_argument("--reverse-steps", type=int, default=100)
    parser.add_argument("--visual-paths", type=int, default=256)
    parser.add_argument("--heat-terms", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir / "plots").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = input_dir / "teacher_dataset.pt"
    metrics_path = input_dir / "metrics.json"
    run_config_path = input_dir / "run_config.json"
    if not dataset_path.exists():
        raise FileNotFoundError(f"missing dataset artifact: {dataset_path}")

    dataset = torch.load(dataset_path, map_location="cpu")
    metrics = _load_json(metrics_path) if metrics_path.exists() else {}
    run_config = _load_json(run_config_path) if run_config_path.exists() else {}

    heat_terms = args.heat_terms
    if heat_terms is None:
        heat_terms = int(metrics.get("heat_terms", run_config.get("heat_terms", 80)))
    seed = args.seed if args.seed is not None else int(run_config.get("seed", 0))
    reverse_steps = args.reverse_steps
    visual_paths = args.visual_paths

    save_target_vs_generated_plot(
        dataset,
        outdir=output_dir,
        n_reverse_steps=reverse_steps,
        n_visual_paths=visual_paths,
        n_heat_terms=heat_terms,
        seed=seed,
    )

    print(f"wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
