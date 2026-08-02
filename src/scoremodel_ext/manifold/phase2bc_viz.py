"""Visualization helpers for the Phase 2B/C S2 smoke pipeline."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Mapping

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def _setup_axis(axis, *, elev: float = 22.0, azim: float = 38.0) -> None:
    axis.view_init(elev=elev, azim=azim)
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.set_xlim(-1.05, 1.05)
    axis.set_ylim(-1.05, 1.05)
    axis.set_zlim(-1.05, 1.05)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")


def _draw_sphere_wireframe(axis) -> None:
    theta = torch.linspace(0.0, 2.0 * math.pi, 40)
    phi = torch.linspace(0.0, math.pi, 20)
    theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing="ij")
    x = torch.cos(theta_grid) * torch.sin(phi_grid)
    y = torch.sin(theta_grid) * torch.sin(phi_grid)
    z = torch.cos(phi_grid)
    axis.plot_wireframe(
        x.numpy(),
        y.numpy(),
        z.numpy(),
        color="0.86",
        linewidth=0.5,
        alpha=0.5,
        rstride=2,
        cstride=2,
    )


def _scatter_s2(axis, points: torch.Tensor, *, title: str, initial_point: torch.Tensor) -> None:
    _setup_axis(axis)
    _draw_sphere_wireframe(axis)
    axis.scatter(
        points[:, 0].numpy(),
        points[:, 1].numpy(),
        points[:, 2].numpy(),
        s=11,
        alpha=0.82,
        linewidths=0.0,
    )
    axis.scatter(
        [float(initial_point[0])],
        [float(initial_point[1])],
        [float(initial_point[2])],
        c="red",
        s=65,
        marker="*",
    )
    axis.set_title(title)


def plot_training_loss(training_history: Mapping[str, object], output_path: Path) -> None:
    epochs = list(training_history.get("epochs", []))
    train_loss = list(training_history.get("train_loss", []))
    validation_loss = list(training_history.get("validation_loss", []))
    if not epochs:
        epochs = [1]
    if not train_loss:
        train_loss = [float(training_history.get("final_train_loss", float("nan")))]
    if not validation_loss:
        validation_loss = [float(training_history.get("final_validation_loss", float("nan")))]

    figure, axis = plt.subplots(figsize=(7.5, 4.6))
    axis.plot(epochs, train_loss, marker="o", label="train_loss")
    axis.plot(epochs, validation_loss, marker="s", label="validation_loss")
    axis.set_xlabel("epoch")
    axis.set_ylabel("loss")
    axis.set_title("Training loss")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_score_prediction_vs_heat(
    predicted_score: torch.Tensor,
    heat_score: torch.Tensor,
    output_path: Path,
) -> None:
    pred = predicted_score.detach().cpu()
    heat = heat_score.detach().cpu()
    n = pred.shape[0]
    if n > 800:
        pred = pred[:800]
        heat = heat[:800]

    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.1))
    labels = ["x", "y", "z"]
    for index in range(3):
        axis = axes[index]
        axis.scatter(heat[:, index].numpy(), pred[:, index].numpy(), s=10, alpha=0.6)
        combined_min = min(float(heat[:, index].min()), float(pred[:, index].min()))
        combined_max = max(float(heat[:, index].max()), float(pred[:, index].max()))
        axis.plot([combined_min, combined_max], [combined_min, combined_max], "k--", linewidth=1.0)
        axis.set_xlabel(f"heat score ({labels[index]})")
        axis.set_ylabel(f"predicted score ({labels[index]})")
        axis.grid(True, alpha=0.25)
    figure.suptitle("Predicted score vs heat reference")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_reverse_samples_single(
    samples: torch.Tensor,
    *,
    initial_point: torch.Tensor,
    title: str,
    output_path: Path,
) -> None:
    figure = plt.figure(figsize=(7.3, 6.1))
    axis = figure.add_subplot(111, projection="3d")
    _scatter_s2(axis, samples.detach().cpu(), title=title, initial_point=initial_point.detach().cpu())
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_reverse_samples_comparison(
    terminal_samples: torch.Tensor,
    generated_samples: Mapping[str, torch.Tensor],
    *,
    initial_point: torch.Tensor,
    output_path: Path,
) -> None:
    figure = plt.figure(figsize=(12.8, 9.8))
    panels = [
        (terminal_samples.detach().cpu(), "terminal samples"),
        (generated_samples["heat"].detach().cpu(), "reverse: heat"),
        (generated_samples["varadhan"].detach().cpu(), "reverse: varadhan"),
        (generated_samples["trained_malliavin"].detach().cpu(), "reverse: trained malliavin"),
    ]
    for index, (samples, title) in enumerate(panels, start=1):
        axis = figure.add_subplot(2, 2, index, projection="3d")
        _scatter_s2(axis, samples, title=title, initial_point=initial_point.detach().cpu())
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_geodesic_distance_comparison(
    distance_by_method: Mapping[str, torch.Tensor],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    bins = 30
    for method, distances in distance_by_method.items():
        axis.hist(
            distances.detach().cpu().numpy(),
            bins=bins,
            alpha=0.5,
            density=True,
            label=method,
        )
    axis.set_xlabel("geodesic distance to initial point (radians)")
    axis.set_ylabel("density")
    axis.set_title("Reverse sample geodesic-distance comparison")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
