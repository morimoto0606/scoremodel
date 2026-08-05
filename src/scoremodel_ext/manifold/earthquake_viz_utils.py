"""Shared Earthquake visualization utilities for library and CLI callers."""

from __future__ import annotations

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import seaborn as sns


def cartesian_to_latlon(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    lat = np.rad2deg(np.arcsin(np.clip(z, -1.0, 1.0)))
    lon = np.rad2deg(np.arctan2(y, x))
    return lat, lon


def spherical_kde_density_on_grid(
    points: np.ndarray,
    grid_size: int,
    kappa: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat = np.linspace(-90.0, 90.0, grid_size // 2)
    lon = np.linspace(-180.0, 180.0, grid_size)
    lat_mesh, lon_mesh = np.meshgrid(lat, lon)

    lat_r = np.deg2rad(lat_mesh.reshape(-1))
    lon_r = np.deg2rad(lon_mesh.reshape(-1))
    grid_xyz = np.stack(
        [
            np.cos(lat_r) * np.cos(lon_r),
            np.cos(lat_r) * np.sin(lon_r),
            np.sin(lat_r),
        ],
        axis=1,
    )

    logits = kappa * (grid_xyz @ points.T)
    logits -= logits.max(axis=1, keepdims=True)
    density = np.exp(logits).mean(axis=1)
    density = density.reshape((lat.shape[0], lon.shape[0]), order="F")
    density /= np.max(density) + 1e-12
    return density, lat, lon


def add_earth_background(ax) -> None:
    """Draw the shared Earth background used by scatter and density plots."""

    ax.set_global()
    ax.set_facecolor("white")
    ax.add_feature(cfeature.OCEAN, zorder=0, facecolor="white")
    ax.add_feature(cfeature.LAND, zorder=1, facecolor="#e0e0e0")


def density_overlay_cmap(
    *,
    max_alpha: float = 0.82,
    n_colors: int = 256,
) -> ListedColormap:
    """Return a density colormap that keeps the Earth visible underneath."""

    colors = plt.get_cmap("cubehelix")(np.linspace(0.0, 1.0, n_colors))
    colors[:, 3] = max_alpha * np.linspace(0.0, 1.0, n_colors) ** 0.8
    return ListedColormap(colors, name="earthquake_density_overlay")


def scatter_earthquake_points(ax, points: np.ndarray, *, role: str) -> None:
    """Scatter Earthquake points with the styles used by De Bortoli et al."""

    styles = {
        "generated": {"size": 1.0, "color_index": 1, "alpha": 1.0},
        "train": {"size": 0.2, "color_index": 5, "alpha": 0.2},
        "test": {"size": 0.2, "color_index": 0, "alpha": 0.2},
    }
    if role not in styles:
        raise ValueError(f"unknown Earthquake scatter role: {role!r}")

    style = styles[role]
    colors = sns.color_palette("hls", 8)
    lat, lon = cartesian_to_latlon(points)
    projected = ax.projection.transform_points(ccrs.Geodetic(), lon, lat)
    ax.scatter(
        projected[:, 0],
        projected[:, 1],
        s=style["size"],
        c=[colors[style["color_index"]]],
        alpha=style["alpha"],
    )


def plot_density_map(
    density: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(5, 5), dpi=300)
    ax = fig.add_subplot(
        1,
        1,
        1,
        projection=ccrs.Orthographic(70, 30),
        frameon=True,
    )
    add_earth_background(ax)
    contour = ax.contourf(
        lon,
        lat,
        density,
        levels=np.linspace(0.0, 1.0, 220),
        transform=ccrs.PlateCarree(),
        antialiased=True,
        cmap=density_overlay_cmap(),
        extend="both",
        zorder=2,
    )
    ax.set_title(title)
    fig.colorbar(contour, ax=ax, shrink=0.8)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
