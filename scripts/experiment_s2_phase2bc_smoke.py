#!/usr/bin/env python3
"""Phase 2B/C smoke pipeline for fixed-start marginal score learning on S2.

This runner only connects existing functions:
- generate_s2_fixed_start_marginal_teacher_dataset
- train_s2_marginal_score / train_s2_score_model
- build_s2_reference_score_functions
- compare_s2_reverse_generators

No backend formulas are changed.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Mapping

import torch

from scoremodel_ext.manifold.experiment_s2_malliavin_teacher import (
    build_s2_reference_score_functions,
    compare_s2_reverse_generators,
    generate_s2_fixed_start_marginal_teacher_dataset,
    train_s2_marginal_score,
    train_s2_score_model,
)
from scoremodel_ext.manifold.phase2bc_viz import (
    plot_geodesic_distance_comparison,
    plot_reverse_samples_comparison,
    plot_reverse_samples_single,
    plot_score_prediction_vs_heat,
    plot_training_loss,
)
from scoremodel_ext.manifold.s2_malliavin import s2_grw_endpoint, s2_heat_kernel_score


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/s2_phase2bc_smoke"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--train-dataset-path", type=Path, default=None)
    parser.add_argument("--validation-dataset-path", type=Path, default=None)
    parser.add_argument("--terminal-samples-path", type=Path, default=None)

    parser.add_argument("--n-paths", type=int, default=256)
    parser.add_argument("--validation-n-paths", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=8)
    parser.add_argument("--minimum-time", type=float, default=0.05)
    parser.add_argument("--maximum-time", type=float, default=0.3)
    parser.add_argument("--covariance-regularization", type=float, default=1e-6)
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--validation-seed", type=int, default=1)
    parser.add_argument("--reverse-seed", type=int, default=0)
    parser.add_argument("--no-vectorize-jacobian", action="store_true")

    parser.add_argument("--training-target", choices=("skorokhod", "direct_score"), default="skorokhod")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--num-frequencies", type=int, default=8)

    parser.add_argument("--heat-terms", type=int, default=80)
    parser.add_argument("--reverse-terminal-time", type=float, default=0.3)
    parser.add_argument("--reverse-minimum-time", type=float, default=0.05)
    parser.add_argument("--reverse-steps", type=int, default=32)
    parser.add_argument("--n-generated-samples", type=int, default=256)
    return parser.parse_args()


def _to_dtype(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def _resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but torch.cuda.is_available() is False")
    return name


def _score_model_forward(model, training_target: str, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if training_target == "skorokhod":
        return model(t, x)
    return model(t, x)


def _training_target_prediction(model, training_target: str, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if training_target == "skorokhod":
        return model.skorokhod_network(t, x)
    return model(t, x)


def _heat_score_batch(initial_point: torch.Tensor, t: torch.Tensor, x: torch.Tensor, n_terms: int) -> torch.Tensor:
    return torch.stack(
        [
            s2_heat_kernel_score(initial_point, endpoint, float(time_value.detach().cpu()), n_terms=n_terms)
            for time_value, endpoint in zip(t, x)
        ]
    )


def _geodesic_distance(initial_point: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    reference = initial_point.reshape(1, 3)
    reference = reference / torch.linalg.vector_norm(reference, dim=1, keepdim=True)
    normalized_points = points / torch.linalg.vector_norm(points, dim=1, keepdim=True)
    cosine = torch.clamp((normalized_points * reference).sum(dim=1), -1.0, 1.0)
    return torch.arccos(cosine)


def _pairwise_mean_geodesic_distance(samples: torch.Tensor) -> float:
    normalized = samples / torch.linalg.vector_norm(samples, dim=1, keepdim=True)
    cosine = torch.clamp(normalized @ normalized.transpose(0, 1), -1.0, 1.0)
    angles = torch.arccos(cosine)
    n = angles.shape[0]
    if n < 2:
        return 0.0
    mask = ~torch.eye(n, dtype=torch.bool, device=angles.device)
    return float(angles[mask].mean())


def _build_trained_score_function(model, training_target: str):
    def _score_fn(t_batch: torch.Tensor, x_batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return _score_model_forward(model, training_target, t_batch, x_batch)

    return _score_fn


def _build_forward_terminal_samples(
    *,
    initial_point: torch.Tensor,
    n_samples: int,
    n_steps: int,
    terminal_time: float,
    seed: int,
) -> torch.Tensor:
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    generator = torch.Generator(device=initial_point.device)
    generator.manual_seed(seed)
    samples = []
    for _ in range(n_samples):
        noise = torch.randn(
            n_steps,
            3,
            dtype=initial_point.dtype,
            device=initial_point.device,
            generator=generator,
        )
        samples.append(s2_grw_endpoint(initial_point, noise, terminal_time))
    return torch.stack(samples, dim=0)


def _load_dataset(path: Path, *, device: str, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    required = {"initial_point", "time", "endpoint", "skorokhod", "score_target"}
    missing = sorted(required.difference(payload.keys()))
    if missing:
        raise KeyError(f"dataset at {path} is missing keys: {missing}")
    return {key: value.to(device=device, dtype=dtype) for key, value in payload.items()}


def _dataset_checks(dataset: Mapping[str, torch.Tensor], minimum_time: float, maximum_time: float) -> Dict[str, object]:
    time_values = dataset["time"]
    initial_points = dataset["initial_point"]
    score_target = dataset["score_target"]

    initial_residual = torch.linalg.vector_norm(initial_points - initial_points[:1], dim=1)
    return {
        "time_min": float(time_values.min()),
        "time_max": float(time_values.max()),
        "time_in_range": bool(float(time_values.min()) >= minimum_time and float(time_values.max()) <= maximum_time),
        "initial_point_fixed": bool(float(initial_residual.max()) <= 1e-12),
        "score_target_all_finite": bool(torch.isfinite(score_target).all().item()),
    }


def _evaluate_training_metrics(
    *,
    model,
    training_target: str,
    training_dataset: Mapping[str, torch.Tensor],
    validation_dataset: Mapping[str, torch.Tensor],
    n_heat_terms: int,
) -> Dict[str, object]:
    t_train = training_dataset["time"]
    x_train = training_dataset["endpoint"]
    t_val = validation_dataset["time"]
    x_val = validation_dataset["endpoint"]

    target_key = "skorokhod" if training_target == "skorokhod" else "score_target"
    y_train = training_dataset[target_key]
    y_val = validation_dataset[target_key]

    with torch.no_grad():
        train_prediction = _training_target_prediction(model, training_target, t_train, x_train)
        val_prediction = _training_target_prediction(model, training_target, t_val, x_val)
        train_loss = torch.mean((train_prediction - y_train) ** 2)
        validation_loss = torch.mean((val_prediction - y_val) ** 2)

        predicted_score = _score_model_forward(model, training_target, t_val, x_val)
        heat_score = _heat_score_batch(validation_dataset["initial_point"][0], t_val, x_val, n_terms=n_heat_terms)
        heat_mse = torch.mean((predicted_score - heat_score) ** 2)
        cosine = torch.nn.functional.cosine_similarity(predicted_score, heat_score, dim=1, eps=1e-12)
        tangent_residual = (predicted_score * x_val).sum(dim=1).abs()

    return {
        "train_loss": float(train_loss),
        "validation_loss": float(validation_loss),
        "heat_score_mse": float(heat_mse),
        "heat_score_mean_cosine": float(cosine.mean()),
        "max_tangent_residual": float(tangent_residual.max()),
        "predicted_score_validation": predicted_score.detach().cpu(),
        "heat_score_validation": heat_score.detach().cpu(),
        "target_key": target_key,
    }


def main() -> None:
    args = _parse_args()
    if args.reverse_minimum_time < args.minimum_time:
        raise ValueError("reverse_minimum_time must be >= minimum_time to stay in learned time range")
    if args.reverse_minimum_time >= args.reverse_terminal_time:
        raise ValueError("reverse_minimum_time must be < reverse_terminal_time")

    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    dtype = _to_dtype(args.dtype)
    device = _resolve_device(args.device)

    run_config = vars(args).copy()
    run_config["resolved_device"] = device
    run_config_path = outdir / "run_config.json"
    with run_config_path.open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2, default=str)

    started = time.perf_counter()

    dataset_seconds = 0.0
    validation_dataset_seconds = 0.0

    if args.train_dataset_path is not None:
        teacher_dataset_path = args.train_dataset_path.resolve()
        training_dataset = _load_dataset(teacher_dataset_path, device=device, dtype=dtype)
    else:
        teacher_dataset_path = outdir / "teacher_dataset.pt"
        dataset_started = time.perf_counter()
        training_dataset = generate_s2_fixed_start_marginal_teacher_dataset(
            n_paths=args.n_paths,
            n_steps=args.n_steps,
            minimum_time=args.minimum_time,
            maximum_time=args.maximum_time,
            covariance_regularization=args.covariance_regularization,
            device=device,
            dtype=dtype,
            seed=args.training_seed,
            vectorize_jacobian=not args.no_vectorize_jacobian,
        )
        dataset_seconds = time.perf_counter() - dataset_started
        torch.save({key: value.detach().cpu() for key, value in training_dataset.items()}, teacher_dataset_path)

    if args.validation_dataset_path is not None:
        validation_dataset_path = args.validation_dataset_path.resolve()
        validation_dataset = _load_dataset(validation_dataset_path, device=device, dtype=dtype)
    else:
        validation_dataset_path = outdir / "validation_dataset.pt"
        validation_dataset_started = time.perf_counter()
        validation_dataset = generate_s2_fixed_start_marginal_teacher_dataset(
            n_paths=args.validation_n_paths,
            n_steps=args.n_steps,
            minimum_time=args.minimum_time,
            maximum_time=args.maximum_time,
            covariance_regularization=args.covariance_regularization,
            device=device,
            dtype=dtype,
            seed=args.validation_seed,
            vectorize_jacobian=not args.no_vectorize_jacobian,
            initial_point=training_dataset["initial_point"][0],
        )
        validation_dataset_seconds = time.perf_counter() - validation_dataset_started
        torch.save({key: value.detach().cpu() for key, value in validation_dataset.items()}, validation_dataset_path)

    dataset_checks = _dataset_checks(training_dataset, args.minimum_time, args.maximum_time)

    train_started = time.perf_counter()
    if args.training_target == "skorokhod":
        model = train_s2_marginal_score(
            training_dataset,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            hidden=args.hidden,
            n_blocks=args.n_blocks,
            num_frequencies=args.num_frequencies,
            device=device,
        )
    else:
        model = train_s2_score_model(
            training_dataset,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            hidden=args.hidden,
            n_blocks=args.n_blocks,
            num_frequencies=args.num_frequencies,
            device=device,
        )
    train_seconds = time.perf_counter() - train_started

    model_path = outdir / "model.pt"
    torch.save(
        {
            "training_target": args.training_target,
            "state_dict": model.state_dict(),
            "resolved_device": device,
            "dtype": args.dtype,
        },
        model_path,
    )

    training_eval = _evaluate_training_metrics(
        model=model,
        training_target=args.training_target,
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        n_heat_terms=args.heat_terms,
    )

    training_history = {
        "epochs": [args.epochs],
        "train_loss": [training_eval["train_loss"]],
        "validation_loss": [training_eval["validation_loss"]],
        "final_train_loss": training_eval["train_loss"],
        "final_validation_loss": training_eval["validation_loss"],
    }
    training_history_path = outdir / "training_history.json"
    with training_history_path.open("w", encoding="utf-8") as handle:
        json.dump(training_history, handle, indent=2)

    terminal_samples_path = outdir / "terminal_samples.pt"
    if args.terminal_samples_path is not None:
        terminal_payload = torch.load(args.terminal_samples_path.resolve(), map_location="cpu")
        terminal_samples = terminal_payload["terminal_samples"].to(device=device, dtype=dtype)
        terminal_samples_path = args.terminal_samples_path.resolve()
    else:
        terminal_samples = _build_forward_terminal_samples(
            initial_point=training_dataset["initial_point"][0],
            n_samples=args.n_generated_samples,
            n_steps=args.n_steps,
            terminal_time=args.reverse_terminal_time,
            seed=args.reverse_seed,
        )
        torch.save({"terminal_samples": terminal_samples.detach().cpu()}, terminal_samples_path)

    reference_functions = build_s2_reference_score_functions(
        training_dataset["initial_point"][0],
        n_heat_terms=args.heat_terms,
    )
    score_functions = dict(reference_functions)
    score_functions["trained_malliavin"] = _build_trained_score_function(model, args.training_target)

    reverse_started = time.perf_counter()
    reverse = compare_s2_reverse_generators(
        terminal_samples,
        score_functions,
        initial_point=training_dataset["initial_point"][0],
        terminal_time=args.reverse_terminal_time,
        n_steps=args.reverse_steps,
        seed=args.reverse_seed,
        minimum_forward_time=args.reverse_minimum_time,
    )
    reverse_seconds = time.perf_counter() - reverse_started

    generated_samples = reverse["generated_samples"]
    reverse_samples_path = outdir / "reverse_samples.pt"
    torch.save(
        {
            "terminal_samples": terminal_samples.detach().cpu(),
            "generated_samples": {k: v.detach().cpu() for k, v in generated_samples.items()},
            "terminal_samples_path": str(terminal_samples_path),
        },
        reverse_samples_path,
    )

    reverse_metrics: Dict[str, Dict[str, float]] = {}
    distance_by_method: Dict[str, torch.Tensor] = {}
    for name, samples in generated_samples.items():
        distances = _geodesic_distance(training_dataset["initial_point"][0], samples)
        norm_error = (torch.linalg.vector_norm(samples, dim=1) - 1.0).abs()
        reverse_metrics[name] = {
            "mean_geodesic_distance_to_initial": float(distances.mean()),
            "median_geodesic_distance_to_initial": float(torch.median(distances)),
            "rmse_geodesic_distance_to_initial": float(torch.sqrt(torch.mean(distances.square()))),
            "max_geodesic_distance_to_initial": float(distances.max()),
            "max_norm_error": float(norm_error.max()),
            "pairwise_mean_geodesic_distance": _pairwise_mean_geodesic_distance(samples),
        }
        distance_by_method[name] = distances.detach().cpu()

    completeness_checks = {
        "teacher_dataset_generated": teacher_dataset_path.exists(),
        "time_in_range": dataset_checks["time_in_range"],
        "initial_point_fixed": dataset_checks["initial_point_fixed"],
        "score_target_finite": dataset_checks["score_target_all_finite"],
        "training_loss_finite": bool(math.isfinite(training_eval["train_loss"])) and bool(math.isfinite(training_eval["validation_loss"])),
        "model_saved": model_path.exists(),
        "reverse_heat_completed": "heat" in generated_samples,
        "reverse_varadhan_completed": "varadhan" in generated_samples,
        "reverse_trained_completed": "trained_malliavin" in generated_samples,
        "generated_samples_near_unit_sphere": all(v["max_norm_error"] <= 1e-5 for v in reverse_metrics.values()),
    }

    plot_training_loss(training_history, outdir / "training_loss.png")
    plot_score_prediction_vs_heat(
        training_eval["predicted_score_validation"],
        training_eval["heat_score_validation"],
        outdir / "score_prediction_vs_heat.png",
    )
    plot_reverse_samples_single(
        generated_samples["heat"],
        initial_point=training_dataset["initial_point"][0],
        title="reverse samples (heat)",
        output_path=outdir / "reverse_samples_heat.png",
    )
    plot_reverse_samples_single(
        generated_samples["varadhan"],
        initial_point=training_dataset["initial_point"][0],
        title="reverse samples (varadhan)",
        output_path=outdir / "reverse_samples_varadhan.png",
    )
    plot_reverse_samples_single(
        generated_samples["trained_malliavin"],
        initial_point=training_dataset["initial_point"][0],
        title="reverse samples (trained malliavin)",
        output_path=outdir / "reverse_samples_trained_malliavin.png",
    )
    plot_reverse_samples_comparison(
        terminal_samples,
        generated_samples,
        initial_point=training_dataset["initial_point"][0],
        output_path=outdir / "reverse_samples_comparison.png",
    )
    plot_geodesic_distance_comparison(distance_by_method, outdir / "geodesic_distance_comparison.png")

    total_seconds = time.perf_counter() - started
    metrics = {
        "training_target": args.training_target,
        "dataset": {
            "n_paths": args.n_paths,
            "validation_n_paths": args.validation_n_paths,
            "n_steps": args.n_steps,
            "minimum_time": args.minimum_time,
            "maximum_time": args.maximum_time,
            "covariance_regularization": args.covariance_regularization,
            "training_seed": args.training_seed,
            "validation_seed": args.validation_seed,
            "checks": dataset_checks,
            "training_dataset_path": str(teacher_dataset_path),
            "validation_dataset_path": str(validation_dataset_path),
            "train_dataset_loaded": bool(args.train_dataset_path is not None),
            "validation_dataset_loaded": bool(args.validation_dataset_path is not None),
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "hidden": args.hidden,
            "n_blocks": args.n_blocks,
            "num_frequencies": args.num_frequencies,
            "train_loss": training_eval["train_loss"],
            "validation_loss": training_eval["validation_loss"],
            "heat_score_mse": training_eval["heat_score_mse"],
            "heat_score_mean_cosine": training_eval["heat_score_mean_cosine"],
            "max_tangent_residual": training_eval["max_tangent_residual"],
            "target_key": training_eval["target_key"],
        },
        "reverse": {
            "terminal_time": args.reverse_terminal_time,
            "minimum_time": args.reverse_minimum_time,
            "reverse_steps": args.reverse_steps,
            "n_generated_samples": args.n_generated_samples,
            "reverse_seed": args.reverse_seed,
            "evaluation_note": "All reverse distance metrics are measured at reverse minimum time, not at t=0.",
            "by_method": reverse_metrics,
            "pairwise_between_methods": reverse["metrics"]["pairwise_mean_geodesic_distance"],
        },
        "timing_seconds": {
            "dataset_generation": dataset_seconds,
            "validation_dataset_generation": validation_dataset_seconds,
            "training": train_seconds,
            "reverse": reverse_seconds,
            "total": total_seconds,
        },
        "training_seed": args.training_seed,
        "validation_seed": args.validation_seed,
        "reverse_seed": args.reverse_seed,
        "validation_n_paths": args.validation_n_paths,
        "reverse_minimum_time": args.reverse_minimum_time,
        "reverse_terminal_time": args.reverse_terminal_time,
        "terminal_samples_path": str(terminal_samples_path),
        "smoke_completion_checks": completeness_checks,
    }

    metrics_path = outdir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print("artifact paths")
    print(f"teacher_dataset: {teacher_dataset_path}")
    print(f"validation_dataset: {validation_dataset_path}")
    print(f"terminal_samples: {terminal_samples_path}")
    print(f"model: {model_path}")
    print(f"training_history: {training_history_path}")
    print(f"reverse_samples: {reverse_samples_path}")
    print(f"metrics: {metrics_path}")
    print(f"run_config: {run_config_path}")
    print(json.dumps(metrics["smoke_completion_checks"], indent=2))


if __name__ == "__main__":
    main()
