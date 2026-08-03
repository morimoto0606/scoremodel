#!/usr/bin/env python3
"""Reproduce earthquake-style S2 density using the S2 Malliavin pipeline.

Outputs are written under results/s2_malliavin_earthquake_reproduction.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import seaborn as sns
import torch

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from scoremodel_ext.manifold.earthquake_adapter import (
    S2TeacherProvider,
    evaluate_s2_score_model,
    load_earthquake_points,
    nearest_neighbor_geodesic_summary,
    s2_rbf_mmd,
)
from scoremodel_ext.manifold.experiment_s2_malliavin_teacher import train_s2_score_model
from scoremodel_ext.manifold.s2_malliavin import s2_reverse_grw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--earthquake-csv",
        type=Path,
        default=Path("upstream/riemannian-score-sde/data/quakes_all.csv"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/s2_malliavin_earthquake_reproduction"),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-paths", type=int, default=512)
    parser.add_argument("--n-steps", type=int, default=10)
    parser.add_argument("--time", type=float, default=0.25)
    parser.add_argument("--gamma-reg", type=float, default=1e-6)
    parser.add_argument("--teacher", choices=("malliavin", "heat", "varadhan"), default="malliavin")
    parser.add_argument("--heat-terms", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--reverse-steps", type=int, default=80)
    parser.add_argument("--n-generate", type=int, default=4096)
    parser.add_argument("--grid-size", type=int, default=400)
    parser.add_argument("--kappa", type=float, default=80.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    return parser.parse_args()


def uniform_s2_samples(n: int, *, rng: np.random.Generator) -> np.ndarray:
    x = rng.normal(size=(n, 3))
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x


def cartesian_to_latlon(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    lat = np.rad2deg(np.arcsin(np.clip(z, -1.0, 1.0)))
    lon = np.rad2deg(np.arctan2(y, x))
    return lat, lon


def spherical_kde_density_on_grid(points: np.ndarray, grid_size: int, kappa: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat = np.linspace(-90.0, 90.0, grid_size // 2)
    lon = np.linspace(-180.0, 180.0, grid_size)
    lat_mesh, lon_mesh = np.meshgrid(lat, lon)

    lat_r = np.deg2rad(lat_mesh.reshape(-1))
    lon_r = np.deg2rad(lon_mesh.reshape(-1))

    gx = np.cos(lat_r) * np.cos(lon_r)
    gy = np.cos(lat_r) * np.sin(lon_r)
    gz = np.sin(lat_r)
    grid_xyz = np.stack([gx, gy, gz], axis=1)

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


def density_overlay_cmap(*, max_alpha: float = 0.82, n_colors: int = 256) -> ListedColormap:
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


def plot_density_map(density: np.ndarray, lat: np.ndarray, lon: np.ndarray, title: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(5, 5), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(70, 30), frameon=True)
    add_earth_background(ax)

    levels = np.linspace(0.0, 1.0, 220)
    contour = ax.contourf(
        lon,
        lat,
        density,
        levels=levels,
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


def plot_scatter_map(
    points: np.ndarray,
    title: str,
    out_path: Path,
    *,
    role: str = "generated",
) -> None:
    fig = plt.figure(figsize=(5, 5), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(70, 30), frameon=True)
    add_earth_background(ax)

    scatter_earthquake_points(ax, points, role=role)
    ax.set_title(title)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _compute_train_validation_indices(
    n_total: int,
    validation_fraction: float,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if n_total < 2:
        raise ValueError("at least two earthquake points are required for train/validation split")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")

    permutation = rng.permutation(n_total)
    n_validation = int(round(validation_fraction * n_total))
    n_validation = max(1, min(n_validation, n_total - 1))
    validation_index = permutation[:n_validation]
    train_index = permutation[n_validation:]
    return train_index, validation_index


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    rng = np.random.default_rng(args.seed)
    all_points_t = load_earthquake_points(args.earthquake_csv, dtype=dtype, device=device)
    all_points = all_points_t.detach().cpu().numpy()

    train_index, validation_index = _compute_train_validation_indices(
        all_points_t.shape[0],
        args.validation_fraction,
        rng=rng,
    )
    train_points_t = all_points_t[train_index]
    validation_points_t = all_points_t[validation_index]

    provider = S2TeacherProvider(
        train_points_t,
        n_steps=args.n_steps,
        covariance_regularization=args.gamma_reg,
        n_heat_terms=args.heat_terms,
        vectorize_jacobian=True,
    )
    teacher_dataset = provider.sample_dataset(
        min(args.n_paths, train_points_t.shape[0]),
        teacher=args.teacher,
        minimum_time=args.time,
        maximum_time=args.time,
        seed=args.seed,
    )
    validation_provider = S2TeacherProvider(
        validation_points_t,
        n_steps=args.n_steps,
        covariance_regularization=args.gamma_reg,
        n_heat_terms=args.heat_terms,
        vectorize_jacobian=True,
    )
    validation_dataset = validation_provider.sample_dataset(
        min(args.n_paths, validation_points_t.shape[0]),
        teacher=args.teacher,
        minimum_time=args.time,
        maximum_time=args.time,
        seed=args.seed + 1,
    )

    train_started = time.perf_counter()
    score_model = train_s2_score_model(
        teacher_dataset,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
    )
    train_seconds = time.perf_counter() - train_started
    score_model.eval()

    train_loss = evaluate_s2_score_model(score_model, teacher_dataset)
    validation_loss = evaluate_s2_score_model(score_model, validation_dataset)

    terminal_points = uniform_s2_samples(args.n_generate, rng=rng)
    terminal_points_t = torch.tensor(terminal_points, dtype=dtype, device=device)

    def score_fn(t_batch: torch.Tensor, x_batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return score_model(t_batch, x_batch)

    generated_t = s2_reverse_grw(
        terminal_points_t,
        score_fn,
        terminal_time=args.time,
        n_steps=args.reverse_steps,
    )
    generated = generated_t.detach().cpu().numpy()
    generated_t_cpu = generated_t.detach().cpu()

    np.save(args.outdir / "generated_samples.npy", generated)
    np.save(args.outdir / "target_samples.npy", all_points)

    target_density, lat, lon = spherical_kde_density_on_grid(all_points, args.grid_size, args.kappa)
    generated_density, _, _ = spherical_kde_density_on_grid(generated, args.grid_size, args.kappa)

    plot_density_map(target_density, lat, lon, "Target Earthquake KDE", args.outdir / "pdf_target_earthquake.png")
    plot_density_map(generated_density, lat, lon, "S2 Malliavin Generated KDE", args.outdir / "pdf_generated_s2_malliavin.png")
    plot_scatter_map(
        all_points,
        "Target Earthquake Points",
        args.outdir / "scatter_target_earthquake.png",
        role="train",
    )
    plot_scatter_map(
        generated,
        "Generated S2 Malliavin Samples",
        args.outdir / "scatter_generated_s2_malliavin.png",
        role="generated",
    )

    # Side-by-side density comparison
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 5),
        subplot_kw={"projection": ccrs.Orthographic(70, 30)},
        dpi=300,
    )
    for ax, density, title in [
        (axes[0], target_density, "Target KDE"),
        (axes[1], generated_density, "Generated KDE (S2 Malliavin)"),
    ]:
        add_earth_background(ax)
        ax.contourf(
            lon,
            lat,
            density,
            levels=np.linspace(0.0, 1.0, 220),
            transform=ccrs.PlateCarree(),
            cmap=density_overlay_cmap(),
            extend="both",
            zorder=2,
        )
        ax.set_title(title)
    plt.tight_layout()
    fig.savefig(args.outdir / "pdf_compare_target_vs_generated.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "device": device,
        "dtype": args.dtype,
        "teacher": args.teacher,
        "n_paths": int(teacher_dataset["initial_point"].shape[0]),
        "n_steps": args.n_steps,
        "reverse_steps": args.reverse_steps,
        "epochs": args.epochs,
        "n_generate": args.n_generate,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "training_seconds": train_seconds,
        "mmd_rbf": s2_rbf_mmd(generated_t_cpu, all_points_t, seed=args.seed),
        "geodesic_distance": nearest_neighbor_geodesic_summary(generated_t_cpu, all_points_t, seed=args.seed),
        "target_resultant": float(np.linalg.norm(all_points.mean(axis=0))),
        "generated_resultant": float(np.linalg.norm(generated.mean(axis=0))),
    }
    with (args.outdir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print("saved", args.outdir / "pdf_generated_s2_malliavin.png")
    print("saved", args.outdir / "pdf_compare_target_vs_generated.png")


if __name__ == "__main__":
    main()
