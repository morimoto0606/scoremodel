#!/usr/bin/env python3
"""Locate the first tensor difference between two reverse-GRW debug traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


TENSOR_ORDER = (
    "input_points",
    "forward_time",
    "time_batch",
    "raw_score",
    "projector",
    "projected_score",
    "raw_noise",
    "projected_noise",
    "tangent_increment",
    "output_points",
    "dt",
    "sqrt_dt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("debug_dir_a", type=Path)
    parser.add_argument("debug_dir_b", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def compare_debug_directories(left_dir: Path, right_dir: Path) -> dict:
    comparisons = []
    first_difference = None
    for step in (0, 1):
        filename = f"reverse_debug_step_{step:03d}.pt"
        left_path = left_dir / filename
        right_path = right_dir / filename
        if not left_path.is_file() or not right_path.is_file():
            missing = {
                "step": step,
                "tensor_name": None,
                "missing_left": not left_path.is_file(),
                "missing_right": not right_path.is_file(),
            }
            comparisons.append(missing)
            if first_difference is None:
                first_difference = missing
            continue
        left_payload = torch.load(left_path, map_location="cpu")
        right_payload = torch.load(right_path, map_location="cpu")
        for name in TENSOR_ORDER:
            left = left_payload.get(name)
            right = right_payload.get(name)
            if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
                row = {
                    "step": step,
                    "tensor_name": name,
                    "missing_left": not isinstance(left, torch.Tensor),
                    "missing_right": not isinstance(right, torch.Tensor),
                }
                different = True
            else:
                same_shape = tuple(left.shape) == tuple(right.shape)
                if same_shape:
                    difference = torch.abs(
                        left.to(torch.float64) - right.to(torch.float64)
                    )
                    max_abs_error = float(difference.max())
                    mean_abs_error = float(difference.mean())
                    exact_equal = bool(torch.equal(left, right))
                else:
                    max_abs_error = None
                    mean_abs_error = None
                    exact_equal = False
                row = {
                    "step": step,
                    "tensor_name": name,
                    "max_abs_error": max_abs_error,
                    "mean_abs_error": mean_abs_error,
                    "left_shape": list(left.shape),
                    "right_shape": list(right.shape),
                    "left_dtype": str(left.dtype),
                    "right_dtype": str(right.dtype),
                    "exact_equal": exact_equal,
                }
                different = not exact_equal
            comparisons.append(row)
            if different and first_difference is None:
                first_difference = row
    return {
        "debug_dir_a": str(left_dir.resolve()),
        "debug_dir_b": str(right_dir.resolve()),
        "first_differing_step": (
            first_difference.get("step") if first_difference else None
        ),
        "first_differing_tensor_name": (
            first_difference.get("tensor_name") if first_difference else None
        ),
        "first_difference": first_difference,
        "comparisons": comparisons,
    }


def main() -> None:
    args = parse_args()
    result = compare_debug_directories(args.debug_dir_a, args.debug_dir_b)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
