"""Generate and diagnose De Bortoli S2 Malliavin teacher data.

Run this module on the GPU server.  It intentionally uses the exact discrete
Skorokhod divergence first; a Hutchinson/VJP implementation can be substituted
after this reference path is validated.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Sequence

import torch

from scoremodel_ext.malliavin.models import train_mirafzali_skorokhod_net

from .s2_malliavin import (
    S2SkorokhodScoreModel,
    s2_heat_kernel_score,
    s2_reverse_grw,
    s2_varadhan_score,
    sample_s2_teacher_path,
)


ScoreFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _knn_conditional_mean(
    endpoints: torch.Tensor,
    weights: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Simple leave-one-out kNN estimate of E[weight | endpoint]."""

    cosine = endpoints @ endpoints.transpose(0, 1)
    distance = 1.0 - cosine
    distance.fill_diagonal_(float("inf"))
    neighbours = torch.topk(distance, k=k, largest=False, dim=1).indices
    return weights[neighbours].mean(dim=1)


def generate_s2_teacher_dataset(
    *,
    n_paths: int,
    n_steps: int,
    terminal_time: float,
    covariance_regularization: float,
    device: str,
    dtype: torch.dtype,
    seed: int,
    vectorize_jacobian: bool,
) -> Dict[str, torch.Tensor]:
    """Generate fixed-start S2 transition paths and full Malliavin weights."""

    if n_paths < 1:
        raise ValueError("n_paths must be positive")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    x0 = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)

    endpoints: List[torch.Tensor] = []
    score_weights: List[torch.Tensor] = []
    directional_weights: List[torch.Tensor] = []
    skorokhod: List[torch.Tensor] = []
    covariance_eigenvalues: List[torch.Tensor] = []
    noises: List[torch.Tensor] = []

    for _ in range(n_paths):
        noise, teacher = sample_s2_teacher_path(
            x0,
            terminal_time=terminal_time,
            n_steps=n_steps,
            covariance_regularization=covariance_regularization,
            generator=generator,
            vectorize_jacobian=vectorize_jacobian,
        )
        noises.append(noise.detach())
        endpoints.append(teacher.endpoint.detach())
        score_weights.append(teacher.score_weight.detach())
        directional_weights.append(teacher.directional_score_weight.detach())
        skorokhod.append(teacher.skorokhod.detach())
        covariance_eigenvalues.append(teacher.covariance_eigenvalues.detach())

    return {
        "initial_point": x0.detach(),
        "time": torch.full((n_paths,), terminal_time, dtype=dtype, device=device),
        "noise": torch.stack(noises),
        "endpoint": torch.stack(endpoints),
        "score_target": torch.stack(score_weights),
        "score_weight": torch.stack(score_weights),
        "directional_score_weight": torch.stack(directional_weights),
        "skorokhod": torch.stack(skorokhod),
        "covariance_eigenvalues": torch.stack(covariance_eigenvalues),
    }


def generate_s2_marginal_teacher_dataset(
    initial_points: torch.Tensor,
    terminal_times: torch.Tensor,
    *,
    n_steps: int,
    covariance_regularization: float = 1e-6,
    seed: int = 0,
    vectorize_jacobian: bool = True,
) -> Dict[str, torch.Tensor]:
    """Generate Algorithm-6 training triples from arbitrary S2 data.

    Sampling ``X_0`` from the data distribution and regressing the returned
    ``skorokhod`` target on ``(t, X_t)`` estimates the *marginal* score.  This
    is the direct De Bortoli base-manifold analogue of Mirafzali Algorithm 6.
    """

    if initial_points.ndim != 2 or initial_points.shape[1] != 3:
        raise ValueError("initial_points must have shape [n_paths, 3]")
    if terminal_times.shape != (initial_points.shape[0],):
        raise ValueError("terminal_times must have shape [n_paths]")
    if bool((terminal_times <= 0).any().detach().cpu()):
        raise ValueError("all terminal_times must be positive")

    generator = torch.Generator(device=initial_points.device)
    generator.manual_seed(seed)
    endpoints = []
    deltas = []
    score_weights = []
    eigenvalues = []
    noises = []
    normalized_initial = initial_points / torch.linalg.vector_norm(
        initial_points, dim=1, keepdim=True
    )

    for initial_point, terminal_time in zip(normalized_initial, terminal_times):
        noise, teacher = sample_s2_teacher_path(
            initial_point,
            terminal_time=float(terminal_time.detach().cpu()),
            n_steps=n_steps,
            covariance_regularization=covariance_regularization,
            generator=generator,
            vectorize_jacobian=vectorize_jacobian,
        )
        noises.append(noise.detach())
        endpoints.append(teacher.endpoint.detach())
        deltas.append(teacher.skorokhod.detach())
        score_weights.append(teacher.score_weight.detach())
        eigenvalues.append(teacher.covariance_eigenvalues.detach())

    return {
        "initial_point": normalized_initial.detach(),
        "time": terminal_times.detach(),
        "noise": torch.stack(noises),
        "endpoint": torch.stack(endpoints),
        "skorokhod": torch.stack(deltas),
        "score_target": torch.stack(score_weights),
        "score_weight": torch.stack(score_weights),
        "covariance_eigenvalues": torch.stack(eigenvalues),
    }


def generate_s2_mixture_marginal_teacher_dataset(
    component_points: torch.Tensor,
    component_weights: torch.Tensor,
    *,
    n_paths: int,
    n_steps: int,
    minimum_time: float,
    maximum_time: float,
    covariance_regularization: float,
    seed: int,
    vectorize_jacobian: bool = True,
) -> Dict[str, torch.Tensor]:
    """Generate variable-time S2 teacher data from a discrete initial mixture."""

    if component_points.ndim != 2 or component_points.shape[1] != 3:
        raise ValueError("component_points must have shape [n_components, 3]")
    if component_weights.shape != (component_points.shape[0],):
        raise ValueError("component_weights must have shape [n_components]")
    if n_paths < 1:
        raise ValueError("n_paths must be positive")
    if minimum_time <= 0 or maximum_time <= 0:
        raise ValueError("minimum_time and maximum_time must be positive")
    if minimum_time > maximum_time:
        raise ValueError("minimum_time must be <= maximum_time")

    normalized_components = component_points / torch.linalg.vector_norm(
        component_points,
        dim=1,
        keepdim=True,
    )
    probabilities = component_weights / component_weights.sum()
    generator = torch.Generator(device=component_points.device)
    generator.manual_seed(seed)
    component_index = torch.multinomial(
        probabilities,
        n_paths,
        replacement=True,
        generator=generator,
    )
    initial_points = normalized_components[component_index]
    terminal_times = torch.empty(
        n_paths,
        dtype=component_points.dtype,
        device=component_points.device,
    ).uniform_(minimum_time, maximum_time, generator=generator)
    dataset = generate_s2_marginal_teacher_dataset(
        initial_points,
        terminal_times,
        n_steps=n_steps,
        covariance_regularization=covariance_regularization,
        seed=seed,
        vectorize_jacobian=vectorize_jacobian,
    )
    dataset["component_index"] = component_index
    return dataset


def generate_s2_fixed_start_marginal_teacher_dataset(
    *,
    n_paths: int,
    n_steps: int,
    minimum_time: float,
    maximum_time: float,
    covariance_regularization: float,
    device: str,
    dtype: torch.dtype,
    seed: int,
    vectorize_jacobian: bool,
    initial_point: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """Generate variable-time S2 teacher triples from one fixed initial point."""

    if n_paths < 1:
        raise ValueError("n_paths must be positive")
    if minimum_time <= 0 or maximum_time <= 0:
        raise ValueError("minimum_time and maximum_time must be positive")
    if minimum_time > maximum_time:
        raise ValueError("minimum_time must be <= maximum_time")

    if initial_point is None:
        initial_point = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
    else:
        initial_point = initial_point.to(device=device, dtype=dtype)
    initial_point = initial_point.reshape(1, 3)
    initial_point = initial_point / torch.linalg.vector_norm(initial_point, dim=1, keepdim=True)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    terminal_times = torch.empty(n_paths, dtype=dtype, device=device).uniform_(minimum_time, maximum_time, generator=generator)
    initial_points = initial_point.expand(n_paths, -1).clone()
    return generate_s2_marginal_teacher_dataset(
        initial_points,
        terminal_times,
        n_steps=n_steps,
        covariance_regularization=covariance_regularization,
        seed=seed,
        vectorize_jacobian=vectorize_jacobian,
    )


def train_s2_marginal_score(
    dataset: Dict[str, torch.Tensor],
    *,
    n_epochs: int = 1000,
    batch_size: int = 2048,
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-5,
    hidden: int = 512,
    n_blocks: int = 6,
    num_frequencies: int = 16,
    device: str = "cuda",
    return_history: bool = False,
) -> S2SkorokhodScoreModel:
    """Train Mirafzali Algorithm 6 and return an intrinsic S2 score model."""

    required = {"time", "endpoint", "skorokhod"}
    missing = required.difference(dataset)
    if missing:
        raise KeyError(f"dataset is missing fields: {sorted(missing)}")
    result = train_mirafzali_skorokhod_net(
        dataset["time"],
        dataset["endpoint"],
        dataset["skorokhod"],
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=learning_rate,
        weight_decay=weight_decay,
        hidden=hidden,
        n_blocks=n_blocks,
        num_frequencies=num_frequencies,
        device=device,
        return_history=return_history,
    )
    if return_history:
        delta_model, history = result
        return S2SkorokhodScoreModel(delta_model), history
    return S2SkorokhodScoreModel(result)


def train_s2_score_model(
    dataset: Dict[str, torch.Tensor],
    *,
    n_epochs: int = 1000,
    batch_size: int = 2048,
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-5,
    hidden: int = 512,
    n_blocks: int = 6,
    num_frequencies: int = 16,
    device: str = "cuda",
    return_history: bool = False,
):
    """Train a direct S2 score regressor with the same architecture and optimizer."""

    required = {"time", "endpoint", "score_target"}
    missing = required.difference(dataset)
    if missing:
        raise KeyError(f"dataset is missing fields: {sorted(missing)}")
    return train_mirafzali_skorokhod_net(
        dataset["time"],
        dataset["endpoint"],
        dataset["score_target"],
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=learning_rate,
        weight_decay=weight_decay,
        hidden=hidden,
        n_blocks=n_blocks,
        num_frequencies=num_frequencies,
        device=device,
        return_history=return_history,
    )


def _normalize_s2_points(points: torch.Tensor) -> torch.Tensor:
    if points.ndim == 1:
        points = points.reshape(1, 3)
    return points / torch.linalg.vector_norm(points, dim=-1, keepdim=True)


def _s2_geodesic_distance(reference_point: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    reference = _normalize_s2_points(reference_point).reshape(1, 3)
    normalized_points = _normalize_s2_points(points)
    cosine = torch.clamp((normalized_points * reference).sum(dim=-1), -1.0, 1.0)
    return torch.arccos(cosine)


def _s2_pairwise_geodesic_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    normalized_left = _normalize_s2_points(left)
    normalized_right = _normalize_s2_points(right)
    cosine = torch.clamp((normalized_left * normalized_right).sum(dim=-1), -1.0, 1.0)
    return torch.arccos(cosine)


def _score_error_metrics(candidate: torch.Tensor, reference: torch.Tensor, *, prefix: str) -> Dict[str, float]:
    errors = candidate - reference
    cosine = torch.nn.functional.cosine_similarity(candidate, reference, dim=1, eps=1e-12)
    return {
        f"{prefix}_rmse": float(torch.sqrt(torch.mean(errors.square()))),
        f"{prefix}_mean_cosine": float(cosine.mean()),
    }


def _default_geodesic_bin_edges(distances: torch.Tensor, n_bins: int = 4) -> torch.Tensor:
    max_distance = float(distances.max())
    if max_distance <= 0:
        max_distance = math.pi
    return torch.linspace(0.0, max_distance, n_bins + 1, dtype=distances.dtype, device=distances.device)


def _binned_score_metrics(
    distances: torch.Tensor,
    metric_pairs: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    bin_edges: torch.Tensor,
) -> List[Dict[str, float | int | None]]:
    bins: List[Dict[str, float | int | None]] = []
    last_index = bin_edges.numel() - 2
    for index, (start, end) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        if index == last_index:
            mask = (distances >= start) & (distances <= end)
        else:
            mask = (distances >= start) & (distances < end)
        entry: Dict[str, float | int | None] = {
            "start_radians": float(start),
            "end_radians": float(end),
            "start_degrees": float(torch.rad2deg(start)),
            "end_degrees": float(torch.rad2deg(end)),
            "count": int(mask.sum()),
        }
        for name, (candidate, reference) in metric_pairs.items():
            if int(mask.sum()) == 0:
                entry[f"{name}_rmse"] = None
                entry[f"{name}_mean_cosine"] = None
                continue
            metrics = _score_error_metrics(candidate[mask], reference[mask], prefix=name)
            entry[f"{name}_rmse"] = metrics[f"{name}_rmse"]
            entry[f"{name}_mean_cosine"] = metrics[f"{name}_mean_cosine"]
        bins.append(entry)
    return bins


def summarize_s2_score_comparison(
    dataset: Dict[str, torch.Tensor],
    *,
    n_heat_terms: int,
    knn_k: int,
    geodesic_bin_edges: Sequence[float] | torch.Tensor | None = None,
) -> Dict[str, object]:
    """Compare Malliavin, heat-kernel, and Varadhan scores on fixed-start S2 data."""

    endpoints = dataset["endpoint"]
    weights = dataset["score_weight"]
    x0 = dataset["initial_point"]
    terminal_time = float(dataset["time"][0])
    n_paths = endpoints.shape[0]
    if not 1 <= knn_k < n_paths:
        raise ValueError("knn_k must satisfy 1 <= knn_k < n_paths")

    malliavin_score = _knn_conditional_mean(endpoints, weights, knn_k)
    heat_score = torch.stack(
        [
            s2_heat_kernel_score(x0, endpoint, terminal_time, n_terms=n_heat_terms)
            for endpoint in endpoints
        ]
    )
    varadhan_score = torch.stack(
        [s2_varadhan_score(x0, endpoint, terminal_time) for endpoint in endpoints]
    )
    geodesic_distance = _s2_geodesic_distance(x0, endpoints)

    if geodesic_bin_edges is None:
        bin_edges = _default_geodesic_bin_edges(geodesic_distance)
    elif isinstance(geodesic_bin_edges, torch.Tensor):
        bin_edges = geodesic_bin_edges.to(device=endpoints.device, dtype=endpoints.dtype)
    else:
        bin_edges = torch.tensor(geodesic_bin_edges, dtype=endpoints.dtype, device=endpoints.device)
    if bin_edges.ndim != 1 or bin_edges.numel() < 2:
        raise ValueError("geodesic_bin_edges must define at least one bin")
    max_distance = geodesic_distance.max()
    if bool((bin_edges[1:] <= bin_edges[:-1]).any().detach().cpu()):
        raise ValueError("geodesic_bin_edges must be strictly increasing")
    if float(bin_edges[-1]) < float(max_distance):
        bin_edges = torch.cat((bin_edges, max_distance.reshape(1)))

    tangent_residual = (endpoints * weights).sum(dim=1).abs()
    endpoint_norm_error = (torch.linalg.vector_norm(endpoints, dim=1) - 1.0).abs()
    eig = dataset["covariance_eigenvalues"]
    metric_pairs = {
        "malliavin_vs_heat": (malliavin_score, heat_score),
        "varadhan_vs_heat": (varadhan_score, heat_score),
        "malliavin_vs_varadhan": (malliavin_score, varadhan_score),
    }

    summary: Dict[str, object] = {
        "n_paths": int(n_paths),
        "terminal_time": terminal_time,
        "knn_k": int(knn_k),
        "mean_score_weight_norm": float(torch.linalg.vector_norm(weights, dim=1).mean()),
        "max_tangent_residual": float(tangent_residual.max()),
        "max_endpoint_norm_error": float(endpoint_norm_error.max()),
        "mean_smallest_covariance_eigenvalue": float(eig[:, 0].mean()),
        "mean_second_covariance_eigenvalue": float(eig[:, 1].mean()),
        "mean_largest_covariance_eigenvalue": float(eig[:, 2].mean()),
        "nan_rate": float((~torch.isfinite(weights)).any(dim=1).double().mean()),
        "mean_geodesic_distance": float(geodesic_distance.mean()),
        "max_geodesic_distance": float(geodesic_distance.max()),
        "geodesic_bins": _binned_score_metrics(
            geodesic_distance,
            metric_pairs,
            bin_edges=bin_edges,
        ),
    }
    for name, (candidate, reference) in metric_pairs.items():
        summary.update(_score_error_metrics(candidate, reference, prefix=name))
    summary["rmse_knn_vs_heat"] = summary["malliavin_vs_heat_rmse"]
    summary["mean_cosine_knn_vs_heat"] = summary["malliavin_vs_heat_mean_cosine"]
    return summary


def diagnose_s2_teacher_dataset(
    dataset: Dict[str, torch.Tensor],
    *,
    n_heat_terms: int,
    knn_k: int,
) -> Dict[str, float]:
    """Compare a kNN conditional mean with the spectral transition score."""
    return summarize_s2_score_comparison(
        dataset,
        n_heat_terms=n_heat_terms,
        knn_k=knn_k,
    )


def _build_heat_score_function(
    initial_point: torch.Tensor,
    *,
    n_heat_terms: int,
):
    """Return score_fn(t, x_batch) for reverse GRW diagnostics."""

    def _score_fn(t_batch: torch.Tensor, x_batch: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                s2_heat_kernel_score(
                    initial_point,
                    endpoint,
                    float(t.detach().cpu()),
                    n_terms=n_heat_terms,
                )
                for t, endpoint in zip(t_batch, x_batch)
            ]
        )

    return _score_fn


def _build_varadhan_score_function(initial_point: torch.Tensor):
    """Return score_fn(t, x_batch) using the Varadhan approximation."""

    def _score_fn(t_batch: torch.Tensor, x_batch: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                s2_varadhan_score(
                    initial_point,
                    endpoint,
                    float(t.detach().cpu()),
                )
                for t, endpoint in zip(t_batch, x_batch)
            ]
        )

    return _score_fn


def build_s2_reference_score_functions(
    initial_point: torch.Tensor,
    *,
    n_heat_terms: int,
) -> Dict[str, ScoreFunction]:
    """Return the analytic S2 score references used in Phase 2."""

    return {
        "heat": _build_heat_score_function(initial_point, n_heat_terms=n_heat_terms),
        "varadhan": _build_varadhan_score_function(initial_point),
    }


def generate_s2_reverse_samples(
    terminal_points: torch.Tensor,
    score_functions: Mapping[str, ScoreFunction],
    *,
    terminal_time: float,
    n_steps: int,
    seed: int,
    minimum_forward_time: float = 1e-3,
) -> Dict[str, torch.Tensor]:
    """Generate reverse GRW samples for several score functions under shared noise."""

    generator = torch.Generator(device=terminal_points.device)
    generator.manual_seed(seed)
    reverse_noise = torch.randn(
        n_steps,
        terminal_points.shape[0],
        3,
        dtype=terminal_points.dtype,
        device=terminal_points.device,
        generator=generator,
    )
    samples = {}
    for name, score_fn in score_functions.items():
        samples[name] = s2_reverse_grw(
            terminal_points,
            score_fn,
            terminal_time=terminal_time,
            n_steps=n_steps,
            standard_noise=reverse_noise,
            minimum_forward_time=minimum_forward_time,
        )
    return samples


def diagnose_s2_reverse_samples(
    generated_samples: Mapping[str, torch.Tensor],
    *,
    initial_point: torch.Tensor,
) -> Dict[str, object]:
    """Summarize reverse-generated S2 samples against the fixed target point."""

    by_method: Dict[str, Dict[str, float]] = {}
    for name, samples in generated_samples.items():
        distance = _s2_geodesic_distance(initial_point, samples)
        norm_error = (torch.linalg.vector_norm(samples, dim=1) - 1.0).abs()
        by_method[name] = {
            "mean_geodesic_distance": float(distance.mean()),
            "rmse_geodesic_distance": float(torch.sqrt(torch.mean(distance.square()))),
            "max_geodesic_distance": float(distance.max()),
            "max_norm_error": float(norm_error.max()),
        }

    pairwise: Dict[str, Dict[str, float]] = {}
    names = list(generated_samples)
    for index, left_name in enumerate(names):
        left = generated_samples[left_name]
        pairwise[left_name] = {}
        for right_name in names[index + 1 :]:
            right = generated_samples[right_name]
            distance = _s2_pairwise_geodesic_distance(left, right)
            pairwise[left_name][right_name] = float(distance.mean())
    return {
        "by_method": by_method,
        "pairwise_mean_geodesic_distance": pairwise,
    }


def compare_s2_reverse_generators(
    terminal_points: torch.Tensor,
    score_functions: Mapping[str, ScoreFunction],
    *,
    initial_point: torch.Tensor,
    terminal_time: float,
    n_steps: int,
    seed: int,
    minimum_forward_time: float = 1e-3,
) -> Dict[str, object]:
    """Run reverse GRW with shared noise and return samples plus comparison metrics."""

    generated_samples = generate_s2_reverse_samples(
        terminal_points,
        score_functions,
        terminal_time=terminal_time,
        n_steps=n_steps,
        seed=seed,
        minimum_forward_time=minimum_forward_time,
    )
    return {
        "generated_samples": generated_samples,
        "metrics": diagnose_s2_reverse_samples(
            generated_samples,
            initial_point=initial_point,
        ),
    }


def save_target_vs_generated_plot(
    dataset: Dict[str, torch.Tensor],
    *,
    outdir: Path,
    n_reverse_steps: int,
    n_visual_paths: int,
    n_heat_terms: int,
    seed: int,
) -> None:
    """Save Stage A target-vs-generated plots for quick quality checks."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping target_vs_generated plots")
        return

    endpoints = dataset["endpoint"]
    x0 = dataset["initial_point"]
    terminal_time = float(dataset["time"][0])
    n_total = endpoints.shape[0]
    if n_total < 1:
        return

    n_samples = min(max(1, n_visual_paths), n_total)
    generator = torch.Generator(device=endpoints.device)
    generator.manual_seed(seed)
    if n_samples == n_total:
        sample_indices = torch.arange(n_total, device=endpoints.device)
    else:
        sample_indices = torch.randperm(n_total, generator=generator, device=endpoints.device)[:n_samples]
    terminal_points = endpoints[sample_indices]

    score_fn = _build_heat_score_function(x0, n_heat_terms=n_heat_terms)
    reverse_noise = torch.randn(
        n_reverse_steps,
        terminal_points.shape[0],
        3,
        dtype=terminal_points.dtype,
        device=terminal_points.device,
        generator=generator,
    )
    generated = s2_reverse_grw(
        terminal_points,
        score_fn,
        terminal_time=terminal_time,
        n_steps=n_reverse_steps,
        standard_noise=reverse_noise,
    )

    endpoints_cpu = terminal_points.detach().cpu()
    generated_cpu = generated.detach().cpu()
    x0_cpu = x0.detach().cpu()
    target_cpu = x0_cpu.repeat(generated_cpu.shape[0], 1)

    figure = plt.figure(figsize=(10, 5))
    ax_target = figure.add_subplot(1, 2, 1, projection="3d")
    ax_generated = figure.add_subplot(1, 2, 2, projection="3d")
    ax_target.scatter(
        target_cpu[:, 0],
        target_cpu[:, 1],
        target_cpu[:, 2],
        s=10,
        alpha=0.8,
        label="target (x0)",
    )
    ax_target.set_title("Target at t=0")
    ax_generated.scatter(
        generated_cpu[:, 0],
        generated_cpu[:, 1],
        generated_cpu[:, 2],
        s=10,
        alpha=0.8,
        label="generated (reverse)",
    )
    ax_generated.set_title("Generated via reverse (heat score)")

    for axis in (ax_target, ax_generated):
        axis.scatter([x0_cpu[0]], [x0_cpu[1]], [x0_cpu[2]], c="red", s=60)
        axis.set_xlim(-1.05, 1.05)
        axis.set_ylim(-1.05, 1.05)
        axis.set_zlim(-1.05, 1.05)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")

    figure.tight_layout()
    figure.savefig(outdir / "target_vs_generated.png", dpi=180)
    plt.close(figure)

    dot_forward = (endpoints_cpu * x0_cpu).sum(dim=1).clamp(-1.0, 1.0)
    dot_generated = (generated_cpu * x0_cpu).sum(dim=1).clamp(-1.0, 1.0)
    forward_angle = torch.rad2deg(torch.arccos(dot_forward))
    generated_angle = torch.rad2deg(torch.arccos(dot_generated))

    figure = plt.figure(figsize=(7, 4))
    plt.hist(forward_angle.numpy(), bins=30, alpha=0.6, label="forward endpoint")
    plt.hist(generated_angle.numpy(), bins=30, alpha=0.6, label="generated")
    plt.xlabel("Angle from target x0 (deg)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    figure.savefig(outdir / "target_vs_generated_angle_hist.png", dpi=180)
    plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-paths", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=8)
    parser.add_argument(
        "--terminal-time",
        "--time",
        dest="terminal_time",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--covariance-regularization",
        "--gamma-reg",
        dest="covariance_regularization",
        type=float,
        default=1e-6,
    )
    parser.add_argument("--heat-terms", type=int, default=80)
    parser.add_argument("--knn-k", type=int, default=8)
    parser.add_argument("--reverse-steps", type=int, default=100)
    parser.add_argument("--visual-paths", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--no-vectorize-jacobian", action="store_true")
    parser.add_argument(
        "--output-dir",
        "--outdir",
        dest="output_dir",
        type=Path,
        default=Path("results/s2_malliavin_teacher"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("experiment start")
    print(f"device: {device}")
    print(f"dtype: {args.dtype}")
    print(f"seed: {args.seed}")
    print(f"n_paths: {args.n_paths}")
    print(f"n_steps: {args.n_steps}")
    print(f"terminal_time: {args.terminal_time}")

    total_started = time.perf_counter()
    print("dataset generation start")
    generation_started = time.perf_counter()
    dataset = generate_s2_teacher_dataset(
        n_paths=args.n_paths,
        n_steps=args.n_steps,
        terminal_time=args.terminal_time,
        covariance_regularization=args.covariance_regularization,
        device=device,
        dtype=dtype,
        seed=args.seed,
        vectorize_jacobian=not args.no_vectorize_jacobian,
    )
    generation_seconds = time.perf_counter() - generation_started
    print("dataset generation completed")

    print("metric computation start")
    metric_started = time.perf_counter()
    metrics = diagnose_s2_teacher_dataset(
        dataset,
        n_heat_terms=args.heat_terms,
        knn_k=args.knn_k,
    )
    metric_seconds = time.perf_counter() - metric_started
    total_seconds = time.perf_counter() - total_started
    print("metric computation completed")

    metrics["n_steps"] = args.n_steps
    metrics["n_paths"] = args.n_paths
    metrics["terminal_time"] = args.terminal_time
    metrics["covariance_regularization"] = args.covariance_regularization
    metrics["heat_terms"] = args.heat_terms
    metrics["knn_k"] = args.knn_k
    metrics["seed"] = args.seed
    metrics["generation_seconds"] = generation_seconds
    metrics["metric_seconds"] = metric_seconds
    metrics["total_seconds"] = total_seconds
    metrics["device"] = device
    metrics["dtype"] = args.dtype

    cpu_dataset = {key: value.cpu() for key, value in dataset.items()}
    teacher_dataset_path = args.output_dir / "teacher_dataset.pt"
    metrics_path = args.output_dir / "metrics.json"
    run_config_path = args.output_dir / "run_config.json"
    torch.save(cpu_dataset, teacher_dataset_path)
    with metrics_path.open("w") as file:
        json.dump(metrics, file, indent=2, sort_keys=True)

    run_config = vars(args).copy()
    run_config["output_dir"] = str(args.output_dir)
    run_config["resolved_device"] = device
    with run_config_path.open("w") as file:
        json.dump(run_config, file, indent=2, default=str)
    with (args.output_dir / "config.json").open("w") as file:
        json.dump(run_config, file, indent=2, default=str)

    save_target_vs_generated_plot(
        dataset,
        outdir=args.output_dir,
        n_reverse_steps=args.reverse_steps,
        n_visual_paths=args.visual_paths,
        n_heat_terms=args.heat_terms,
        seed=args.seed,
    )

    print("artifact paths")
    print(f"teacher_dataset: {teacher_dataset_path}")
    print(f"metrics: {metrics_path}")
    print(f"run_config: {run_config_path}")
    print(f"total elapsed seconds: {total_seconds:.6f}")

    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
