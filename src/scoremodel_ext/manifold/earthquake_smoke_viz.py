"""Visualization helpers for Earthquake teacher-comparison smoke runs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import cartopy.crs as ccrs

from scripts.reproduce_earthquake_s2_malliavin import (
    add_earth_background,
    cartesian_to_latlon,
    density_overlay_cmap,
    plot_density_map,
    scatter_earthquake_points,
    spherical_kde_density_on_grid,
)


DENSITY_TEACHER_ORDER = ("heat", "varadhan", "malliavin")
SCATTER_PANEL_ORDER = ("observed", *DENSITY_TEACHER_ORDER)


def _to_numpy(points: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(points, torch.Tensor):
        return points.detach().cpu().numpy()
    return points


def _stable_subsample(points: np.ndarray, n: int) -> np.ndarray:
    if points.shape[0] <= n:
        return points
    return points[:n]


def generate_earthquake_scatter_comparison(
    *,
    observed_points: torch.Tensor | np.ndarray,
    generated_by_teacher: Mapping[str, torch.Tensor | np.ndarray],
    output_path: Path,
    max_points: int = 4096,
    marker_size: float = 1.0,
    alpha: float = 0.65,
    view_lon: float = 70.0,
    view_lat: float = 30.0,
    save_pdf: bool = True,
) -> dict:
    """Save one fixed-projection Observed/Heat/Varadhan/Malliavin scatter figure."""

    missing = [
        teacher
        for teacher in DENSITY_TEACHER_ORDER
        if teacher not in generated_by_teacher
    ]
    if missing:
        raise ValueError(f"missing generated samples for teachers: {missing}")
    if max_points < 1:
        raise ValueError("max_points must be positive")

    panels = {"observed": _to_numpy(observed_points)}
    panels.update(
        {
            teacher: _to_numpy(generated_by_teacher[teacher])
            for teacher in DENSITY_TEACHER_ORDER
        }
    )
    for name, points in panels.items():
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"{name} points must have shape (n, 3), got {points.shape}")
        if not np.isfinite(points).all():
            raise ValueError(f"{name} points contain non-finite values")

    displayed_count = min(max_points, *(points.shape[0] for points in panels.values()))
    if displayed_count < 1:
        raise ValueError("all panels must contain at least one point")
    displayed = {
        name: _stable_subsample(points, displayed_count)
        for name, points in panels.items()
    }

    titles = {
        "observed": "Observed",
        "heat": "Heat",
        "varadhan": "Varadhan",
        "malliavin": "Malliavin",
    }
    projection = ccrs.Orthographic(view_lon, view_lat)
    fig = plt.figure(figsize=(20, 5), dpi=300)
    for index, panel_name in enumerate(SCATTER_PANEL_ORDER, start=1):
        ax = fig.add_subplot(
            1,
            4,
            index,
            projection=projection,
            frameon=True,
        )
        add_earth_background(ax)
        lat, lon = cartesian_to_latlon(displayed[panel_name])
        projected = ax.projection.transform_points(ccrs.Geodetic(), lon, lat)
        ax.scatter(
            projected[:, 0],
            projected[:, 1],
            s=marker_size,
            c="#d95f02",
            alpha=alpha,
        )
        ax.set_title(titles[panel_name])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    pdf_path = output_path.with_suffix(".pdf") if save_pdf else None
    if pdf_path is not None:
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "panel_order": SCATTER_PANEL_ORDER,
        "displayed_count_per_panel": displayed_count,
        "marker_size": marker_size,
        "alpha": alpha,
        "output_path": output_path,
        "pdf_path": pdf_path,
    }


def generate_earthquake_density_plots(
    *,
    observed_train_points: torch.Tensor | np.ndarray,
    observed_validation_points: torch.Tensor | np.ndarray,
    generated_by_teacher: Mapping[str, torch.Tensor | np.ndarray],
    output_dir: Path,
    grid_size: int = 400,
    kappa: float = 80.0,
    view_lon: float = 70.0,
    view_lat: float = 30.0,
) -> dict:
    """Save individual densities and the fixed-order shared comparison."""

    output_dir.mkdir(parents=True, exist_ok=True)
    observed = np.concatenate(
        (
            _to_numpy(observed_train_points),
            _to_numpy(observed_validation_points),
        ),
        axis=0,
    )
    generated_np = {
        teacher: _to_numpy(points)
        for teacher, points in generated_by_teacher.items()
        if teacher in DENSITY_TEACHER_ORDER
    }
    teachers = [
        teacher for teacher in DENSITY_TEACHER_ORDER if teacher in generated_np
    ]

    density_observed, lat, lon = spherical_kde_density_on_grid(
        observed,
        grid_size,
        kappa,
    )
    per_teacher_density: Dict[str, np.ndarray] = {}
    for teacher in teachers:
        density, _, _ = spherical_kde_density_on_grid(
            generated_np[teacher],
            grid_size,
            kappa,
        )
        per_teacher_density[teacher] = density
        plot_density_map(
            density,
            lat,
            lon,
            f"Generated Density ({teacher})",
            output_dir / f"earthquake_density_{teacher}.png",
        )

    panel_order = ("observed", *teachers)
    levels = np.linspace(0.0, 1.0, 220)
    density_cmap = density_overlay_cmap()
    fig = plt.figure(figsize=(5 * len(panel_order), 5), dpi=300)
    densities = {"observed": density_observed, **per_teacher_density}
    titles = {
        "observed": "Observed Density",
        "heat": "Heat Density",
        "varadhan": "Varadhan Density",
        "malliavin": "Malliavin Density",
    }
    for column, panel_key in enumerate(panel_order, start=1):
        ax = fig.add_subplot(
            1,
            len(panel_order),
            column,
            projection=ccrs.Orthographic(view_lon, view_lat),
            frameon=True,
        )
        add_earth_background(ax)
        ax.contourf(
            lon,
            lat,
            densities[panel_key],
            levels=levels,
            transform=ccrs.PlateCarree(),
            antialiased=True,
            cmap=density_cmap,
            extend="both",
            zorder=2,
        )
        ax.set_title(titles[panel_key])
    fig.tight_layout()
    output_path = output_dir / "earthquake_density_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {
        "panel_order": panel_order,
        "n_columns": len(panel_order),
        "observed_count": int(observed.shape[0]),
        "output_path": output_path,
    }


def generate_earthquake_smoke_plots(
    *,
    observed_points: torch.Tensor | np.ndarray,
    generated_by_teacher: Mapping[str, torch.Tensor | np.ndarray],
    training_history_by_teacher: Mapping[str, Dict[str, list[float]]],
    output_dir: Path,
    time_diagnostics_by_teacher: Mapping[str, Sequence[dict]] | None = None,
    observed_train_points: torch.Tensor | np.ndarray | None = None,
    observed_test_points: torch.Tensor | np.ndarray | None = None,
    grid_size: int = 400,
    kappa: float = 80.0,
    view_lon: float = 70.0,
    view_lat: float = 30.0,
    max_scatter_points: int | None = None,
) -> None:
    """Create required fixed-condition globe/density/loss comparison figures."""

    output_dir.mkdir(parents=True, exist_ok=True)

    observed = _to_numpy(observed_points)
    observed_train = (
        observed if observed_train_points is None else _to_numpy(observed_train_points)
    )
    observed_test = (
        np.empty((0, 3), dtype=observed.dtype)
        if observed_test_points is None
        else _to_numpy(observed_test_points)
    )
    generated_np = {name: _to_numpy(points) for name, points in generated_by_teacher.items()}

    def scatter_points(points: np.ndarray) -> np.ndarray:
        if max_scatter_points is None:
            return points
        return _stable_subsample(points, max_scatter_points)

    def save_scatter_figure(
        path: Path,
        title: str,
        *,
        generated: np.ndarray | None = None,
    ) -> None:
        fig = plt.figure(figsize=(5, 5), dpi=300)
        ax = fig.add_subplot(
            1,
            1,
            1,
            projection=ccrs.Orthographic(view_lon, view_lat),
            frameon=True,
        )
        add_earth_background(ax)
        if generated is not None:
            scatter_earthquake_points(ax, scatter_points(generated), role="generated")
        scatter_earthquake_points(ax, scatter_points(observed_train), role="train")
        if observed_test.shape[0] > 0:
            scatter_earthquake_points(ax, scatter_points(observed_test), role="test")
        ax.set_title(title)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    save_scatter_figure(
        output_dir / "earthquake_observed_globe.png",
        "Observed Earthquake Points",
    )

    for teacher in DENSITY_TEACHER_ORDER:
        if teacher not in generated_np:
            continue
        generated = generated_np[teacher]
        save_scatter_figure(
            output_dir / f"earthquake_generated_{teacher}.png",
            f"Generated Points ({teacher})",
            generated=generated,
        )

    # Generated globe comparison.
    fig = plt.figure(figsize=(15, 5), dpi=300)
    teachers = [name for name in DENSITY_TEACHER_ORDER if name in generated_np]
    for index, teacher in enumerate(teachers, start=1):
        ax = fig.add_subplot(1, len(teachers), index, projection=ccrs.Orthographic(view_lon, view_lat), frameon=True)
        add_earth_background(ax)
        scatter_earthquake_points(
            ax,
            scatter_points(generated_np[teacher]),
            role="generated",
        )
        scatter_earthquake_points(ax, scatter_points(observed_train), role="train")
        if observed_test.shape[0] > 0:
            scatter_earthquake_points(ax, scatter_points(observed_test), role="test")
        ax.set_title(f"Generated ({teacher})")
    fig.suptitle(f"Generated Sample Comparison (view lon={view_lon}, lat={view_lat})")
    fig.tight_layout()
    fig.savefig(output_dir / "earthquake_generated_comparison.png", bbox_inches="tight")
    plt.close(fig)

    generate_earthquake_density_plots(
        observed_train_points=observed_train,
        observed_validation_points=observed_test,
        generated_by_teacher=generated_np,
        output_dir=output_dir,
        grid_size=grid_size,
        kappa=kappa,
        view_lon=view_lon,
        view_lat=view_lat,
    )

    # Time-local target diagnostics.  These use the validation split so all
    # teacher curves are evaluated on the shared held-out time samples.
    if time_diagnostics_by_teacher:
        diagnostic_specs = (
            (
                "target_norm_mean",
                "Mean target norm",
                "Target Norm by Time",
                "target_norm_by_time.png",
            ),
            (
                "time_bin_target_dispersion",
                "Time-bin target dispersion",
                "Time-bin Target Dispersion",
                "time_bin_target_dispersion_by_time.png",
            ),
        )
        for metric_key, ylabel, title, filename in diagnostic_specs:
            fig = plt.figure(figsize=(7, 4), dpi=220)
            ax = fig.add_subplot(1, 1, 1)
            for teacher in ("heat", "varadhan", "malliavin"):
                rows = time_diagnostics_by_teacher.get(teacher)
                if not rows:
                    continue
                x = [float(row["time"]) for row in rows if row.get(metric_key) is not None]
                y = [float(row[metric_key]) for row in rows if row.get(metric_key) is not None]
                if x:
                    ax.plot(x, y, marker="o", linewidth=1.8, label=teacher)
            ax.set_xlabel("Time")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / filename, bbox_inches="tight")
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
