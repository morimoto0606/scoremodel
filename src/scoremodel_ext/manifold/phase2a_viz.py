"""Phase 2A visualizations for the S^2 Malliavin teacher experiments.

This module only consumes saved artifacts and analytic score references. It
does not alter the numerical backend used to generate the dataset.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .s2_malliavin import s2_heat_kernel_score, s2_varadhan_score


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_phase2a_run(input_dir: Path) -> tuple[Dict[str, torch.Tensor], dict, dict]:
    """Load a saved Phase 2A run from ``input_dir``."""

    dataset_path = input_dir / "teacher_dataset.pt"
    metrics_path = input_dir / "metrics.json"
    run_config_path = input_dir / "run_config.json"
    if not dataset_path.exists():
        raise FileNotFoundError(f"missing dataset artifact: {dataset_path}")

    dataset = torch.load(dataset_path, map_location="cpu")
    metrics = _load_json(metrics_path) if metrics_path.exists() else {}
    run_config = _load_json(run_config_path) if run_config_path.exists() else {}
    return dataset, metrics, run_config


def _normalize(points: torch.Tensor) -> torch.Tensor:
    if points.ndim == 1:
        points = points.reshape(1, -1)
    return points / torch.linalg.vector_norm(points, dim=-1, keepdim=True)


def _score_cosine(candidate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(candidate, reference, dim=-1, eps=1e-12)


def _geodesic_distance(reference_point: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    normalized_reference = _normalize(reference_point).reshape(1, 3)
    normalized_points = _normalize(points)
    cosine = torch.clamp((normalized_points * normalized_reference).sum(dim=-1), -1.0, 1.0)
    return torch.arccos(cosine)


def _score_error_norm(candidate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(candidate - reference, dim=-1)


def compute_phase2a_scores(
    dataset: Mapping[str, torch.Tensor],
    metrics: Mapping[str, object],
) -> Dict[str, torch.Tensor]:
    """Compute analytic score references and pointwise diagnostics."""

    endpoints = dataset["endpoint"].detach().cpu()
    malliavin = dataset["score_weight"].detach().cpu()
    initial_point = dataset["initial_point"].detach().cpu()
    terminal_time = float(dataset["time"][0].detach().cpu())
    n_heat_terms = int(metrics.get("heat_terms", 80))

    heat = torch.stack(
        [
            s2_heat_kernel_score(initial_point, endpoint, terminal_time, n_terms=n_heat_terms)
            for endpoint in endpoints
        ]
    ).detach().cpu()
    varadhan = torch.stack(
        [s2_varadhan_score(initial_point, endpoint, terminal_time) for endpoint in endpoints]
    ).detach().cpu()
    geodesic_distance = _geodesic_distance(initial_point, endpoints)

    return {
        "endpoints": endpoints,
        "malliavin": malliavin,
        "heat": heat,
        "varadhan": varadhan,
        "geodesic_distance": geodesic_distance,
        "error_norm": _score_error_norm(malliavin, heat),
        "cosine_similarity": _score_cosine(malliavin, heat),
    }


def _sphere_axes(ax: plt.Axes) -> None:
    theta = torch.linspace(0.0, 2.0 * math.pi, 40)
    phi = torch.linspace(0.0, math.pi, 20)
    theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing="ij")
    x = torch.cos(theta_grid) * torch.sin(phi_grid)
    y = torch.sin(theta_grid) * torch.sin(phi_grid)
    z = torch.cos(phi_grid)
    ax.plot_wireframe(
        x.numpy(),
        y.numpy(),
        z.numpy(),
        color="0.85",
        linewidth=0.5,
        alpha=0.55,
        rstride=2,
        cstride=2,
    )
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(-1.05, 1.05)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def _sample_indices(n_points: int, max_points: int, seed: int) -> torch.Tensor:
    max_points = max(1, min(int(max_points), n_points))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randperm(n_points, generator=generator)[:max_points]


def plot_endpoint_scatter(
    endpoints: torch.Tensor,
    output_path: Path,
    *,
    title: str = "Phase 2A endpoint scatter on S^2",
) -> None:
    figure = plt.figure(figsize=(7.5, 6.5))
    axis = figure.add_subplot(111, projection="3d")
    _sphere_axes(axis)
    axis.scatter(
        endpoints[:, 0].numpy(),
        endpoints[:, 1].numpy(),
        endpoints[:, 2].numpy(),
        s=10,
        c=torch.linalg.vector_norm(endpoints, dim=1).numpy(),
        cmap="viridis",
        alpha=0.8,
        linewidths=0.0,
    )
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_scalar_on_sphere(
    endpoints: torch.Tensor,
    values: torch.Tensor,
    output_path: Path,
    *,
    title: str,
    cmap: str = "viridis",
    colorbar_label: str = "value",
) -> None:
    figure = plt.figure(figsize=(7.5, 6.5))
    axis = figure.add_subplot(111, projection="3d")
    _sphere_axes(axis)
    scatter = axis.scatter(
        endpoints[:, 0].numpy(),
        endpoints[:, 1].numpy(),
        endpoints[:, 2].numpy(),
        c=values.numpy(),
        cmap=cmap,
        s=13,
        alpha=0.95,
        linewidths=0.0,
    )
    axis.set_title(title)
    colorbar = figure.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_score_arrows(
    endpoints: torch.Tensor,
    heat: torch.Tensor,
    malliavin: torch.Tensor,
    output_path: Path,
    *,
    max_points: int = 160,
    seed: int = 0,
    arrow_length: float = 0.18,
) -> None:
    indices = _sample_indices(endpoints.shape[0], max_points=max_points, seed=seed)
    sampled_endpoints = endpoints[indices]
    sampled_heat = heat[indices]
    sampled_malliavin = malliavin[indices]

    figure = plt.figure(figsize=(14.0, 6.5))
    axes = [figure.add_subplot(1, 2, 1, projection="3d"), figure.add_subplot(1, 2, 2, projection="3d")]
    panels = [
        (axes[0], sampled_heat, "Heat score arrows"),
        (axes[1], sampled_malliavin, "Malliavin score arrows"),
    ]
    for axis, vectors, title in panels:
        _sphere_axes(axis)
        axis.scatter(
            sampled_endpoints[:, 0].numpy(),
            sampled_endpoints[:, 1].numpy(),
            sampled_endpoints[:, 2].numpy(),
            s=12,
            c="black",
            alpha=0.35,
            linewidths=0.0,
        )
        axis.quiver(
            sampled_endpoints[:, 0].numpy(),
            sampled_endpoints[:, 1].numpy(),
            sampled_endpoints[:, 2].numpy(),
            vectors[:, 0].numpy(),
            vectors[:, 1].numpy(),
            vectors[:, 2].numpy(),
            length=arrow_length,
            normalize=True,
            color="#1f77b4",
            linewidth=0.8,
            alpha=0.8,
        )
        axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_geodesic_bins(metrics: Mapping[str, object], output_path: Path) -> None:
    bins = list(metrics.get("geodesic_bins", []))
    if not bins:
        raise ValueError("metrics does not contain geodesic_bins")

    centers = []
    rmse_heat = []
    rmse_varadhan = []
    cosine_heat = []
    cosine_varadhan = []
    counts = []
    for entry in bins:
        start = float(entry["start_degrees"])
        end = float(entry["end_degrees"])
        centers.append(0.5 * (start + end))
        counts.append(int(entry["count"]))
        rmse_heat.append(float(entry["malliavin_vs_heat_rmse"]))
        rmse_varadhan.append(float(entry["varadhan_vs_heat_rmse"]))
        cosine_heat.append(float(entry["malliavin_vs_heat_mean_cosine"]))
        cosine_varadhan.append(float(entry["varadhan_vs_heat_mean_cosine"]))

    figure, axes = plt.subplots(2, 1, figsize=(9.5, 8.5), sharex=True)
    axes[0].plot(centers, rmse_heat, marker="o", label="Malliavin vs heat RMSE")
    axes[0].plot(centers, rmse_varadhan, marker="s", label="Varadhan vs heat RMSE")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("Geodesic-bin diagnostics")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(centers, cosine_heat, marker="o", label="Malliavin vs heat cosine")
    axes[1].plot(centers, cosine_varadhan, marker="s", label="Varadhan vs heat cosine")
    axes[1].set_xlabel("Geodesic bin center (degrees)")
    axes[1].set_ylabel("Cosine similarity")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="lower left")

    count_axis = axes[1].twinx()
    count_axis.bar(centers, counts, width=max(1.0, (centers[1] - centers[0]) * 0.75 if len(centers) > 1 else 10.0), alpha=0.12, color="gray")
    count_axis.set_ylabel("Sample count")

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_covariance_eigenvalues(
    covariance_eigenvalues: torch.Tensor,
    output_path: Path,
    *,
    title: str = "Covariance eigenvalue distribution",
) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    labels = ["smallest", "middle", "largest"]
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    for column, label, color in zip(covariance_eigenvalues.T, labels, colors):
        axis.hist(
            column.numpy(),
            bins=40,
            alpha=0.45,
            label=label,
            color=color,
            density=True,
        )
    axis.set_title(title)
    axis.set_xlabel("Eigenvalue")
    axis.set_ylabel("Density")
    axis.legend()
    axis.grid(True, alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_line, separator, *body])


def write_phase2a_tables(
    metrics: Mapping[str, object],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write markdown and CSV summary tables for a Phase 2A run."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "phase2a_summary.md"
    bins_csv_path = output_dir / "phase2a_geodesic_bins.csv"

    summary_keys = [
        "seed",
        "n_paths",
        "n_steps",
        "terminal_time",
        "knn_k",
        "heat_terms",
        "malliavin_vs_heat_rmse",
        "malliavin_vs_heat_mean_cosine",
        "malliavin_vs_varadhan_rmse",
        "varadhan_vs_heat_rmse",
        "max_endpoint_norm_error",
        "max_tangent_residual",
        "nan_rate",
        "generation_seconds",
        "metric_seconds",
        "total_seconds",
    ]
    summary_rows = [[key, metrics.get(key, "")] for key in summary_keys]
    summary_path.write_text(
        _markdown_table(["metric", "value"], summary_rows) + "\n",
        encoding="utf-8",
    )

    bins = list(metrics.get("geodesic_bins", []))
    if bins:
        headers = list(bins[0].keys())
        with bins_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            writer.writerows(bins)
    else:
        bins_csv_path.write_text("", encoding="utf-8")

    return summary_path, bins_csv_path


def generate_phase2a_visuals(
    input_dir: Path,
    output_dir: Path | None = None,
    *,
    max_arrow_points: int = 160,
    arrow_length: float = 0.18,
) -> Dict[str, torch.Tensor]:
    """Render the full Phase 2A visual bundle from saved artifacts."""

    dataset, metrics, run_config = load_phase2a_run(input_dir)
    output_dir = (output_dir or input_dir / "plots").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = compute_phase2a_scores(dataset, metrics)
    seed = int(run_config.get("seed", metrics.get("seed", 0)))

    plot_endpoint_scatter(summary["endpoints"], output_dir / "endpoint_scatter_s2.png")
    plot_score_arrows(
        summary["endpoints"],
        summary["heat"],
        summary["malliavin"],
        output_dir / "heat_vs_malliavin_arrows.png",
        max_points=max_arrow_points,
        seed=seed,
        arrow_length=arrow_length,
    )
    plot_scalar_on_sphere(
        summary["endpoints"],
        summary["error_norm"],
        output_dir / "error_norm_on_sphere.png",
        title="Malliavin vs heat error norm on S^2",
        cmap="magma",
        colorbar_label="||score_M - score_heat||",
    )
    plot_scalar_on_sphere(
        summary["endpoints"],
        summary["cosine_similarity"],
        output_dir / "cosine_similarity_on_sphere.png",
        title="Malliavin vs heat cosine similarity on S^2",
        cmap="viridis",
        colorbar_label="cosine similarity",
    )
    plot_geodesic_bins(metrics, output_dir / "geodesic_bin_rmse.png")
    plot_covariance_eigenvalues(
        dataset["covariance_eigenvalues"].detach().cpu(),
        output_dir / "covariance_eigenvalues_distribution.png",
    )
    write_phase2a_tables(metrics, output_dir)

    return summary
