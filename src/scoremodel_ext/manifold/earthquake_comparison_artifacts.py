"""Saved-artifact loading shared by Earthquake comparison CLIs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


TEACHERS = ("heat", "varadhan", "malliavin")
DEFAULT_PREFIX = "earthquake_linear_beta_100k_ema"
DEFAULT_COMPARISON_DIR = Path(f"results/{DEFAULT_PREFIX}_comparison")
UPSTREAM_ANTIPODAL_COORDINATES = "upstream-earthquake-antipodal"
STANDARD_EARTH_COORDINATES = "standard-earth"
SUPPORTED_EARTH_COORDINATES = (
    UPSTREAM_ANTIPODAL_COORDINATES,
    STANDARD_EARTH_COORDINATES,
)


def load_tensor_artifact(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"missing artifact: {path}")
    value = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected tensor artifact at {path}, got {type(value).__name__}")
    return value


def load_upstream_generated_samples(
    path: Path,
    *,
    coordinate_system: str | None = None,
) -> torch.Tensor:
    """Load upstream reverse samples and convert them to standard Earth xyz.

    The upstream Earthquake adapter embeds every geographic point at the
    antipode of the convention used by :mod:`scoremodel_ext`.  Conversion is
    intentionally never inferred from the filename: callers must either pass
    ``coordinate_system`` or provide a sibling JSON metadata file containing
    that field.
    """

    resolved_path = path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"missing upstream sample artifact: {resolved_path}")
    if resolved_path.suffix == ".npy":
        value = np.load(resolved_path, allow_pickle=False)
        if not isinstance(value, np.ndarray):
            raise TypeError(f"expected ndarray artifact at {resolved_path}")
        points = torch.from_numpy(value)
    elif resolved_path.suffix == ".pt":
        points = load_tensor_artifact(resolved_path)
    else:
        raise ValueError(
            "upstream samples must be stored as .npy or .pt: "
            f"{resolved_path}"
        )

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            "upstream samples must have shape (n, 3), got "
            f"{tuple(points.shape)}"
        )
    if points.shape[0] < 1:
        raise ValueError("upstream samples must not be empty")
    if not bool(torch.isfinite(points).all()):
        raise ValueError("upstream samples contain non-finite values")

    metadata_path = resolved_path.with_suffix(".json")
    if coordinate_system is None and metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if not isinstance(metadata, dict):
            raise TypeError(f"expected JSON object in metadata: {metadata_path}")
        coordinate_system = metadata.get("coordinate_system")
    if coordinate_system is None:
        raise ValueError(
            "upstream coordinate system is unknown; provide a sibling JSON "
            "metadata file or --upstream-coordinate-system"
        )
    if coordinate_system not in SUPPORTED_EARTH_COORDINATES:
        raise ValueError(f"unknown upstream coordinate system: {coordinate_system!r}")

    points = points.detach().clone()
    if coordinate_system == UPSTREAM_ANTIPODAL_COORDINATES:
        points.neg_()
    norms = torch.linalg.vector_norm(points, dim=1, keepdim=True)
    if bool((norms <= 0).any()):
        raise ValueError("upstream samples contain a zero vector")
    maximum_norm_error = torch.max(torch.abs(norms - 1.0))
    if bool(maximum_norm_error > 1e-4):
        raise ValueError(
            "upstream samples are not on the unit sphere: "
            f"maximum norm error={float(maximum_norm_error):.6g}"
        )
    return points


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
