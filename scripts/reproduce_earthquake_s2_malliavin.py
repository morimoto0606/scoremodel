#!/usr/bin/env python3
"""Reproduce earthquake-style S2 density using the S2 Malliavin pipeline.

Outputs are written under results/s2_malliavin_earthquake_reproduction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from scoremodel_ext.manifold.experiment_s2_malliavin_teacher import (
    generate_s2_marginal_teacher_dataset,
    train_s2_marginal_score,
)
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
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--reverse-steps", type=int, default=80)
    parser.add_argument("--n-generate", type=int, default=4096)
    parser.add_argument("--grid-size", type=int, default=400)
    parser.add_argument("--kappa", type=float, default=80.0)
    return parser.parse_args()


def load_earthquake_points(csv_path: Path) -> np.ndarray:
    latlon_deg = np.genfromtxt(csv_path, delimiter=",", skip_header=4)
    lat = np.deg2rad(latlon_deg[:, 0])
    lon = np.deg2rad(latlon_deg[:, 1])
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    points = np.stack([x, y, z], axis=1)
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    return points


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


def plot_density_map(density: np.ndarray, lat: np.ndarray, lon: np.ndarray, title: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(5, 5), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(70, 30), frameon=True)
    ax.set_global()
    ax.add_feature(cfeature.LAND, zorder=0, facecolor="#e0e0e0")

    levels = np.linspace(0.0, 1.0, 220)
    contour = ax.contourf(
        lon,
        lat,
        density,
        levels=levels,
        transform=ccrs.PlateCarree(),
        antialiased=True,
        cmap="cubehelix",
        extend="both",
    )
    alpha_gradient = np.linspace(0.0, 1.0, len(ax.collections))
    for col, alpha in zip(ax.collections, alpha_gradient):
        col.set_alpha(alpha)

    ax.set_title(title)
    fig.colorbar(contour, ax=ax, shrink=0.8)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_map(points: np.ndarray, title: str, out_path: Path) -> None:
    lat, lon = cartesian_to_latlon(points)
    fig = plt.figure(figsize=(5, 5), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(70, 30), frameon=True)
    ax.set_global()
    ax.add_feature(cfeature.LAND, zorder=0, facecolor="#e0e0e0")

    xy = ax.projection.transform_points(ccrs.Geodetic(), lon, lat)
    ax.scatter(xy[:, 0], xy[:, 1], s=0.4, alpha=0.3, c="#1f77b4")
    ax.set_title(title)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    rng = np.random.default_rng(args.seed)
    all_points = load_earthquake_points(args.earthquake_csv)

    sel = rng.choice(all_points.shape[0], size=min(args.n_paths, all_points.shape[0]), replace=False)
    initial_points_np = all_points[sel]
    initial_points = torch.tensor(initial_points_np, dtype=dtype, device=device)
    terminal_times = torch.full((initial_points.shape[0],), args.time, dtype=dtype, device=device)

    teacher_dataset = generate_s2_marginal_teacher_dataset(
        initial_points,
        terminal_times,
        n_steps=args.n_steps,
        covariance_regularization=args.gamma_reg,
        seed=args.seed,
        vectorize_jacobian=True,
    )

    score_model = train_s2_marginal_score(
        teacher_dataset,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
    )
    score_model.eval()

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

    np.save(args.outdir / "generated_samples.npy", generated)
    np.save(args.outdir / "target_samples.npy", all_points)

    target_density, lat, lon = spherical_kde_density_on_grid(all_points, args.grid_size, args.kappa)
    generated_density, _, _ = spherical_kde_density_on_grid(generated, args.grid_size, args.kappa)

    plot_density_map(target_density, lat, lon, "Target Earthquake KDE", args.outdir / "pdf_target_earthquake.png")
    plot_density_map(generated_density, lat, lon, "S2 Malliavin Generated KDE", args.outdir / "pdf_generated_s2_malliavin.png")
    plot_scatter_map(all_points, "Target Earthquake Points", args.outdir / "scatter_target_earthquake.png")
    plot_scatter_map(generated, "Generated S2 Malliavin Samples", args.outdir / "scatter_generated_s2_malliavin.png")

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
        ax.set_global()
        ax.add_feature(cfeature.LAND, zorder=0, facecolor="#e0e0e0")
        ax.contourf(
            lon,
            lat,
            density,
            levels=np.linspace(0.0, 1.0, 220),
            transform=ccrs.PlateCarree(),
            cmap="cubehelix",
            extend="both",
        )
        ax.set_title(title)
    plt.tight_layout()
    fig.savefig(args.outdir / "pdf_compare_target_vs_generated.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "device": device,
        "dtype": args.dtype,
        "n_paths": int(initial_points.shape[0]),
        "n_steps": args.n_steps,
        "reverse_steps": args.reverse_steps,
        "epochs": args.epochs,
        "n_generate": args.n_generate,
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
