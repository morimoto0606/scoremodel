"""Visualization helpers for Earthquake teacher-comparison smoke runs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from scripts.reproduce_earthquake_s2_malliavin import (
    cartesian_to_latlon,
    plot_density_map,
    plot_scatter_map,
    spherical_kde_density_on_grid,
)


def _to_numpy(points: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(points, torch.Tensor):
        return points.detach().cpu().numpy()
    return points


def _stable_subsample(points: np.ndarray, n: int) -> np.ndarray:
    if points.shape[0] <= n:
        return points
    return points[:n]


def generate_earthquake_smoke_plots(
    *,
    observed_points: torch.Tensor | np.ndarray,
    generated_by_teacher: Mapping[str, torch.Tensor | np.ndarray],
    training_history_by_teacher: Mapping[str, Dict[str, list[float]]],
    output_dir: Path,
    grid_size: int = 400,
    kappa: float = 80.0,
    view_lon: float = 70.0,
    view_lat: float = 30.0,
    max_scatter_points: int = 5000,
) -> None:
    """Create required fixed-condition globe/density/loss comparison figures."""

    output_dir.mkdir(parents=True, exist_ok=True)

    observed = _to_numpy(observed_points)
    generated_np = {name: _to_numpy(points) for name, points in generated_by_teacher.items()}

    observed_scatter = _stable_subsample(observed, max_scatter_points)
    plot_scatter_map(
        observed_scatter,
        "Observed Earthquake Points",
        output_dir / "earthquake_observed_globe.png",
    )

    density_observed, lat, lon = spherical_kde_density_on_grid(observed, grid_size, kappa)

    per_teacher_density: Dict[str, np.ndarray] = {}
    for teacher in ("heat", "varadhan", "malliavin"):
        if teacher not in generated_np:
            continue
        generated = generated_np[teacher]
        generated_scatter = _stable_subsample(generated, max_scatter_points)
        plot_scatter_map(
            generated_scatter,
            f"Generated Points ({teacher})",
            output_dir / f"earthquake_generated_{teacher}.png",
        )

        density, _, _ = spherical_kde_density_on_grid(generated, grid_size, kappa)
        per_teacher_density[teacher] = density
        plot_density_map(
            density,
            lat,
            lon,
            f"Generated Density ({teacher})",
            output_dir / f"earthquake_density_{teacher}.png",
        )

    # Generated globe comparison.
    fig = plt.figure(figsize=(15, 5), dpi=300)
    teachers = [name for name in ("heat", "varadhan", "malliavin") if name in generated_np]
    for index, teacher in enumerate(teachers, start=1):
        ax = fig.add_subplot(1, len(teachers), index, projection=ccrs.Orthographic(view_lon, view_lat), frameon=True)
        ax.set_global()
        ax.add_feature(cfeature.LAND, zorder=0, facecolor="#e0e0e0")
        lat_v, lon_v = cartesian_to_latlon(_stable_subsample(generated_np[teacher], max_scatter_points))
        xy = ax.projection.transform_points(ccrs.Geodetic(), lon_v, lat_v)
        ax.scatter(xy[:, 0], xy[:, 1], s=0.3, alpha=0.35, c="#1f77b4")
        ax.set_title(f"Generated ({teacher})")
    fig.suptitle(f"Generated Sample Comparison (view lon={view_lon}, lat={view_lat})")
    fig.tight_layout()
    fig.savefig(output_dir / "earthquake_generated_comparison.png", bbox_inches="tight")
    plt.close(fig)

    # Density comparison with identical observed/generated normalization.
    fig = plt.figure(figsize=(5 * (len(teachers) + 1), 5), dpi=300)
    ax_obs = fig.add_subplot(1, len(teachers) + 1, 1, projection=ccrs.Orthographic(view_lon, view_lat), frameon=True)
    ax_obs.set_global()
    ax_obs.add_feature(cfeature.LAND, zorder=0, facecolor="#e0e0e0")
    levels = np.linspace(0.0, 1.0, 220)
    ax_obs.contourf(
        lon,
        lat,
        density_observed,
        levels=levels,
        transform=ccrs.PlateCarree(),
        cmap="cubehelix",
        extend="both",
    )
    ax_obs.set_title("Observed Density")
    for offset, teacher in enumerate(teachers, start=1):
        ax = fig.add_subplot(
            1,
            len(teachers) + 1,
            offset + 1,
            projection=ccrs.Orthographic(view_lon, view_lat),
            frameon=True,
        )
        ax.set_global()
        ax.add_feature(cfeature.LAND, zorder=0, facecolor="#e0e0e0")
        ax.contourf(
            lon,
            lat,
            per_teacher_density[teacher],
            levels=levels,
            transform=ccrs.PlateCarree(),
            cmap="cubehelix",
            extend="both",
        )
        ax.set_title(f"Generated Density ({teacher})")
    fig.tight_layout()
    fig.savefig(output_dir / "earthquake_density_comparison.png", bbox_inches="tight")
    plt.close(fig)

    # Training-loss comparison.
    fig = plt.figure(figsize=(7, 4), dpi=220)
    ax = fig.add_subplot(1, 1, 1)
    for teacher in ("heat", "varadhan", "malliavin"):
        history = training_history_by_teacher.get(teacher)
        if not history:
            continue
        epochs = history.get("epochs", [])
        losses = history.get("train_loss", [])
        if len(epochs) != len(losses) or len(losses) == 0:
            continue
        ax.plot(epochs, losses, label=teacher, linewidth=2.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss")
    ax.set_title("Training Loss Comparison")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_loss_comparison.png", bbox_inches="tight")
    plt.close(fig)