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
from typing import Dict, List

import torch

from scoremodel_ext.malliavin.models import train_mirafzali_skorokhod_net

from .s2_malliavin import (
    S2SkorokhodScoreModel,
    s2_heat_kernel_score,
    s2_reverse_grw,
    sample_s2_teacher_path,
)


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
        "score_weight": torch.stack(score_weights),
        "covariance_eigenvalues": torch.stack(eigenvalues),
    }


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
) -> S2SkorokhodScoreModel:
    """Train Mirafzali Algorithm 6 and return an intrinsic S2 score model."""

    required = {"time", "endpoint", "skorokhod"}
    missing = required.difference(dataset)
    if missing:
        raise KeyError(f"dataset is missing fields: {sorted(missing)}")
    delta_model = train_mirafzali_skorokhod_net(
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
    )
    return S2SkorokhodScoreModel(delta_model)


def diagnose_s2_teacher_dataset(
    dataset: Dict[str, torch.Tensor],
    *,
    n_heat_terms: int,
    knn_k: int,
) -> Dict[str, float]:
    """Compare a kNN conditional mean with the spectral transition score."""

    endpoints = dataset["endpoint"]
    weights = dataset["score_weight"]
    x0 = dataset["initial_point"]
    terminal_time = float(dataset["time"][0])
    n_paths = endpoints.shape[0]
    if not 1 <= knn_k < n_paths:
        raise ValueError("knn_k must satisfy 1 <= knn_k < n_paths")

    estimated_score = _knn_conditional_mean(endpoints, weights, knn_k)
    reference_score = torch.stack(
        [
            s2_heat_kernel_score(
                x0,
                endpoint,
                terminal_time,
                n_terms=n_heat_terms,
            )
            for endpoint in endpoints
        ]
    )
    errors = estimated_score - reference_score
    tangent_residual = (endpoints * weights).sum(dim=1).abs()
    endpoint_norm_error = (
        torch.linalg.vector_norm(endpoints, dim=1) - 1.0
    ).abs()
    eig = dataset["covariance_eigenvalues"]

    cosine = torch.nn.functional.cosine_similarity(
        estimated_score,
        reference_score,
        dim=1,
        eps=1e-12,
    )
    return {
        "n_paths": int(n_paths),
        "terminal_time": terminal_time,
        "knn_k": int(knn_k),
        "rmse_knn_vs_heat": float(torch.sqrt(torch.mean(errors.square()))),
        "mean_cosine_knn_vs_heat": float(cosine.mean()),
        "mean_score_weight_norm": float(torch.linalg.vector_norm(weights, dim=1).mean()),
        "max_tangent_residual": float(tangent_residual.max()),
        "max_endpoint_norm_error": float(endpoint_norm_error.max()),
        "mean_smallest_covariance_eigenvalue": float(eig[:, 0].mean()),
        "mean_second_covariance_eigenvalue": float(eig[:, 1].mean()),
        "mean_largest_covariance_eigenvalue": float(eig[:, 2].mean()),
        "nan_rate": float((~torch.isfinite(weights)).any(dim=1).double().mean()),
    }


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
    parser.add_argument("--time", type=float, default=0.3)
    parser.add_argument("--gamma-reg", type=float, default=1e-6)
    parser.add_argument("--heat-terms", type=int, default=80)
    parser.add_argument("--knn-k", type=int, default=8)
    parser.add_argument("--reverse-steps", type=int, default=100)
    parser.add_argument("--visual-paths", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--no-vectorize-jacobian", action="store_true")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/s2_malliavin_teacher"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    args.outdir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    dataset = generate_s2_teacher_dataset(
        n_paths=args.n_paths,
        n_steps=args.n_steps,
        terminal_time=args.time,
        covariance_regularization=args.gamma_reg,
        device=args.device,
        dtype=dtype,
        seed=args.seed,
        vectorize_jacobian=not args.no_vectorize_jacobian,
    )
    generation_seconds = time.perf_counter() - started
    metrics = diagnose_s2_teacher_dataset(
        dataset,
        n_heat_terms=args.heat_terms,
        knn_k=args.knn_k,
    )
    metrics["n_steps"] = args.n_steps
    metrics["covariance_regularization"] = args.gamma_reg
    metrics["generation_seconds"] = generation_seconds
    metrics["device"] = args.device
    metrics["dtype"] = args.dtype

    cpu_dataset = {key: value.cpu() for key, value in dataset.items()}
    torch.save(cpu_dataset, args.outdir / "teacher_dataset.pt")
    with (args.outdir / "metrics.json").open("w") as file:
        json.dump(metrics, file, indent=2, sort_keys=True)
    with (args.outdir / "config.json").open("w") as file:
        json.dump(vars(args) | {"outdir": str(args.outdir)}, file, indent=2, default=str)

    save_target_vs_generated_plot(
        dataset,
        outdir=args.outdir,
        n_reverse_steps=args.reverse_steps,
        n_visual_paths=args.visual_paths,
        n_heat_terms=args.heat_terms,
        seed=args.seed,
    )

    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
