"""Saved-artifact loading shared by Earthquake comparison CLIs."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch


TEACHERS = ("heat", "varadhan", "malliavin")
DEFAULT_PREFIX = "earthquake_linear_beta_100k_ema"
DEFAULT_COMPARISON_DIR = Path(f"results/{DEFAULT_PREFIX}_comparison")


def load_tensor_artifact(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"missing artifact: {path}")
    value = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected tensor artifact at {path}, got {type(value).__name__}")
    return value


def load_observed_points(run_dirs: Mapping[str, Path]) -> torch.Tensor:
    """Load observed points saved before teacher generation, including --skip-viz runs."""

    for teacher in TEACHERS:
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
        teacher: load_tensor_artifact(run_dirs[teacher] / "generated_samples.pt")
        for teacher in TEACHERS
    }
    return observed, generated
