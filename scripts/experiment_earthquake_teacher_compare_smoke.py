#!/usr/bin/env python3
"""Earthquake smoke runner with fixed-condition teacher comparison on S2."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import torch

from scoremodel_ext.manifold.earthquake_adapter import (
    load_earthquake_points,
    nearest_neighbor_geodesic_summary,
    s2_rbf_mmd,
)
from scoremodel_ext.manifold.experiment_s2_malliavin_teacher import (
    train_s2_marginal_score,
    train_s2_score_model,
)
from scoremodel_ext.manifold.s2_malliavin import (
    s2_discrete_malliavin_teacher,
    s2_grw_endpoint,
    s2_heat_kernel_score,
    s2_reverse_grw,
    s2_varadhan_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", choices=("heat", "varadhan", "malliavin"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("upstream/riemannian-score-sde/data/quakes_all.csv"),
    )
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--validation-size", type=int, default=128)
    parser.add_argument("--n-steps", type=int, default=8)
    parser.add_argument("--minimum-time", type=float, default=0.05)
    parser.add_argument("--maximum-time", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--num-frequencies", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--reverse-seed", type=int, default=0)
    parser.add_argument("--reverse-steps", type=int, default=32)
    parser.add_argument("--n-generated-samples", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--split-indices-path", type=Path, default=None)
    parser.add_argument("--terminal-samples-path", type=Path, default=None)
    parser.add_argument("--reverse-noise-path", type=Path, default=None)
    parser.add_argument("--covariance-regularization", type=float, default=1e-6)
    parser.add_argument("--heat-terms", type=int, default=80)
    parser.add_argument("--viz-output-dir", type=Path, default=None)
    parser.add_argument("--skip-viz", action="store_true")
    return parser.parse_args()


def to_dtype(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def resolve_device(name: str) -> str:
    if name == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return name


def normalize(points: torch.Tensor) -> torch.Tensor:
    return points / torch.linalg.vector_norm(points, dim=1, keepdim=True).clamp_min(1e-12)


def compute_split_indices(
    n_total: int,
    train_size: int,
    validation_size: int,
    *,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if train_size < 1 or validation_size < 1:
        raise ValueError("train_size and validation_size must be positive")
    if train_size + validation_size > n_total:
        raise ValueError("train_size + validation_size exceeds available earthquake samples")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(n_total, generator=generator)
    train_index = permutation[:train_size]
    validation_index = permutation[train_size : train_size + validation_size]
    return train_index, validation_index


def maybe_load_or_create_split_indices(
    *,
    split_indices_path: Path | None,
    output_dir: Path,
    n_total: int,
    train_size: int,
    validation_size: int,
    split_seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, Path, Path]:
    train_path = output_dir / "train_indices.pt"
    validation_path = output_dir / "validation_indices.pt"

    if split_indices_path is not None:
        root = split_indices_path
        if root.is_file():
            payload = torch.load(root, map_location="cpu")
            train_idx = payload["train_indices"].long()
            validation_idx = payload["validation_indices"].long()
            torch.save(train_idx, train_path)
            torch.save(validation_idx, validation_path)
            return train_idx, validation_idx, train_path, validation_path
        root.mkdir(parents=True, exist_ok=True)
        root_train = root / "train_indices.pt"
        root_validation = root / "validation_indices.pt"
        if root_train.exists() and root_validation.exists():
            train_idx = torch.load(root_train, map_location="cpu").long()
            validation_idx = torch.load(root_validation, map_location="cpu").long()
            torch.save(train_idx, train_path)
            torch.save(validation_idx, validation_path)
            return train_idx, validation_idx, train_path, validation_path

    train_idx, validation_idx = compute_split_indices(
        n_total,
        train_size,
        validation_size,
        seed=split_seed,
    )
    torch.save(train_idx, train_path)
    torch.save(validation_idx, validation_path)
    if split_indices_path is not None:
        root = split_indices_path
        root.mkdir(parents=True, exist_ok=True)
        torch.save(train_idx, root / "train_indices.pt")
        torch.save(validation_idx, root / "validation_indices.pt")
    return train_idx, validation_idx, train_path, validation_path


def maybe_load_or_create_terminal_samples(
    *,
    path: Path | None,
    output_path: Path,
    n_generated_samples: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
) -> torch.Tensor:
    if path is not None and path.exists():
        tensor = torch.load(path, map_location="cpu")
        if isinstance(tensor, dict):
            tensor = tensor["terminal_samples"]
        tensor = tensor.to(device=device, dtype=dtype)
        torch.save(tensor.detach().cpu(), output_path)
        return tensor

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    samples = torch.randn(n_generated_samples, 3, generator=generator, dtype=dtype, device=device)
    samples = normalize(samples)
    torch.save(samples.detach().cpu(), output_path)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(samples.detach().cpu(), path)
    return samples


def maybe_load_or_create_reverse_noise(
    *,
    path: Path | None,
    output_path: Path,
    reverse_steps: int,
    n_generated_samples: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
) -> torch.Tensor:
    if path is not None and path.exists():
        tensor = torch.load(path, map_location="cpu")
        if isinstance(tensor, dict):
            tensor = tensor["reverse_noise"]
        tensor = tensor.to(device=device, dtype=dtype)
        torch.save(tensor.detach().cpu(), output_path)
        return tensor

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    noise = torch.randn(reverse_steps, n_generated_samples, 3, generator=generator, dtype=dtype, device=device)
    torch.save(noise.detach().cpu(), output_path)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(noise.detach().cpu(), path)
    return noise


def build_teacher_dataset(
    *,
    initial_points: torch.Tensor,
    times: torch.Tensor,
    noises: torch.Tensor,
    teacher: str,
    covariance_regularization: float,
    heat_terms: int,
) -> Dict[str, torch.Tensor]:
    endpoints = []
    score_target = []
    skorokhod = []

    for initial_point, terminal_time, noise in zip(initial_points, times, noises):
        terminal_time_float = float(terminal_time.detach().cpu())
        if teacher == "malliavin":
            teacher_state = s2_discrete_malliavin_teacher(
                initial_point,
                noise,
                terminal_time_float,
                covariance_regularization=covariance_regularization,
                vectorize_jacobian=True,
            )
            endpoints.append(teacher_state.endpoint)
            score_target.append(teacher_state.score_weight)
            skorokhod.append(teacher_state.skorokhod)
        elif teacher == "heat":
            endpoint = s2_grw_endpoint(initial_point, noise, terminal_time_float)
            endpoints.append(endpoint)
            score_target.append(
                s2_heat_kernel_score(
                    initial_point,
                    endpoint,
                    terminal_time_float,
                    n_terms=heat_terms,
                )
            )
        else:
            endpoint = s2_grw_endpoint(initial_point, noise, terminal_time_float)
            endpoints.append(endpoint)
            score_target.append(s2_varadhan_score(initial_point, endpoint, terminal_time_float))

    dataset: Dict[str, torch.Tensor] = {
        "initial_point": initial_points,
        "time": times,
        "noise": noises,
        "endpoint": torch.stack(endpoints, dim=0),
        "score_target": torch.stack(score_target, dim=0),
    }
    if teacher == "malliavin":
        dataset["skorokhod"] = torch.stack(skorokhod, dim=0)
    return {
        key: value.detach() if isinstance(value, torch.Tensor) else value
        for key, value in dataset.items()
    }


def evaluate_dataset_loss(
    model,
    dataset: Dict[str, torch.Tensor],
    *,
    teacher: str,
) -> float:
    with torch.no_grad():
        if teacher == "malliavin":
            prediction = model.skorokhod_network(dataset["time"], dataset["endpoint"])
            target = dataset["skorokhod"]
        else:
            prediction = model(dataset["time"], dataset["endpoint"])
            target = dataset["score_target"]
        value = torch.mean((prediction - target) ** 2)
    return float(value)


def build_score_fn(model):
    def _score_fn(time_batch: torch.Tensor, x_batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model(time_batch, x_batch)

    return _score_fn


def generated_norm_error(samples: torch.Tensor) -> float:
    norm = torch.linalg.vector_norm(samples, dim=1)
    return float(torch.mean(torch.abs(norm - 1.0)))


def finite_rate(samples: torch.Tensor) -> float:
    return float(torch.isfinite(samples).all(dim=1).double().mean())


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_log_path = output_dir / "run.log"

    def log(message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line)
        with run_log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    dtype = to_dtype(args.dtype)
    device = resolve_device(args.device)

    if args.teacher != "heat":
        heat_dir = output_dir.parent / "heat"
        if args.split_indices_path is None and (heat_dir / "train_indices.pt").exists() and (heat_dir / "validation_indices.pt").exists():
            args.split_indices_path = heat_dir
        if args.terminal_samples_path is None and (heat_dir / "terminal_samples.pt").exists():
            args.terminal_samples_path = heat_dir / "terminal_samples.pt"
        if args.reverse_noise_path is None and (heat_dir / "reverse_noise.pt").exists():
            args.reverse_noise_path = heat_dir / "reverse_noise.pt"

    run_config = vars(args).copy()
    run_config["resolved_device"] = device
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2, default=str)

    log(f"loading earthquake points from {args.data_path}")
    points = load_earthquake_points(args.data_path, dtype=dtype, device=device)

    train_idx, validation_idx, train_idx_path, validation_idx_path = maybe_load_or_create_split_indices(
        split_indices_path=args.split_indices_path,
        output_dir=output_dir,
        n_total=points.shape[0],
        train_size=args.train_size,
        validation_size=args.validation_size,
        split_seed=args.split_seed,
    )
    log(f"split indices ready: {train_idx_path.name}, {validation_idx_path.name}")

    train_initial = normalize(points[train_idx.to(device=device)])
    validation_initial = normalize(points[validation_idx.to(device=device)])

    time_samples_path = output_dir / "time_samples.pt"
    noise_samples_path = output_dir / "teacher_noises.pt"
    if args.teacher != "heat":
        sibling = output_dir.parent / "heat"
        sibling_time = sibling / "time_samples.pt"
        sibling_noise = sibling / "teacher_noises.pt"
    else:
        sibling_time = None
        sibling_noise = None

    if sibling_time is not None and sibling_time.exists():
        payload = torch.load(sibling_time, map_location="cpu")
        train_times = payload["train_times"].to(device=device, dtype=dtype)
        validation_times = payload["validation_times"].to(device=device, dtype=dtype)
    else:
        time_generator = torch.Generator(device=device)
        time_generator.manual_seed(args.seed)
        train_times = torch.empty(args.train_size, dtype=dtype, device=device).uniform_(
            args.minimum_time,
            args.maximum_time,
            generator=time_generator,
        )
        validation_times = torch.empty(args.validation_size, dtype=dtype, device=device).uniform_(
            args.minimum_time,
            args.maximum_time,
            generator=time_generator,
        )
    torch.save(
        {
            "train_times": train_times.detach().cpu(),
            "validation_times": validation_times.detach().cpu(),
        },
        time_samples_path,
    )

    if sibling_noise is not None and sibling_noise.exists():
        payload = torch.load(sibling_noise, map_location="cpu")
        train_noises = payload["train_noises"].to(device=device, dtype=dtype)
        validation_noises = payload["validation_noises"].to(device=device, dtype=dtype)
    else:
        noise_generator = torch.Generator(device=device)
        noise_generator.manual_seed(args.seed + 11)
        train_noises = torch.randn(args.train_size, args.n_steps, 3, generator=noise_generator, dtype=dtype, device=device)
        validation_noises = torch.randn(
            args.validation_size,
            args.n_steps,
            3,
            generator=noise_generator,
            dtype=dtype,
            device=device,
        )
    torch.save(
        {
            "train_noises": train_noises.detach().cpu(),
            "validation_noises": validation_noises.detach().cpu(),
        },
        noise_samples_path,
    )

    teacher_started = time.perf_counter()
    train_dataset = build_teacher_dataset(
        initial_points=train_initial,
        times=train_times,
        noises=train_noises,
        teacher=args.teacher,
        covariance_regularization=args.covariance_regularization,
        heat_terms=args.heat_terms,
    )
    validation_dataset = build_teacher_dataset(
        initial_points=validation_initial,
        times=validation_times,
        noises=validation_noises,
        teacher=args.teacher,
        covariance_regularization=args.covariance_regularization,
        heat_terms=args.heat_terms,
    )
    teacher_generation_seconds = time.perf_counter() - teacher_started

    train_started = time.perf_counter()
    if args.teacher == "malliavin":
        model, history = train_s2_marginal_score(
            train_dataset,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            hidden=args.hidden,
            n_blocks=args.n_blocks,
            num_frequencies=args.num_frequencies,
            device=device,
            return_history=True,
        )
        training_path = "marginal_skorokhod"
    else:
        model, history = train_s2_score_model(
            train_dataset,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            hidden=args.hidden,
            n_blocks=args.n_blocks,
            num_frequencies=args.num_frequencies,
            device=device,
            return_history=True,
        )
        training_path = "direct_score"
    training_seconds = time.perf_counter() - train_started

    initial_train_loss = float(history.get("initial_train_loss", float("nan")))
    final_train_loss = float(history.get("final_train_loss", float("nan")))
    best_train_loss = float(history.get("best_train_loss", float("nan")))

    validation_loss = evaluate_dataset_loss(model, validation_dataset, teacher=args.teacher)

    terminal_samples = maybe_load_or_create_terminal_samples(
        path=args.terminal_samples_path,
        output_path=output_dir / "terminal_samples.pt",
        n_generated_samples=args.n_generated_samples,
        dtype=dtype,
        device=device,
        seed=args.seed + 101,
    )
    reverse_noise = maybe_load_or_create_reverse_noise(
        path=args.reverse_noise_path,
        output_path=output_dir / "reverse_noise.pt",
        reverse_steps=args.reverse_steps,
        n_generated_samples=args.n_generated_samples,
        dtype=dtype,
        device=device,
        seed=args.reverse_seed,
    )

    reverse_started = time.perf_counter()
    generated = s2_reverse_grw(
        terminal_samples,
        build_score_fn(model),
        terminal_time=args.maximum_time,
        n_steps=args.reverse_steps,
        standard_noise=reverse_noise,
        minimum_forward_time=args.minimum_time,
    )
    reverse_sampling_seconds = time.perf_counter() - reverse_started

    generated_cpu = generated.detach().cpu()
    train_dataset_cpu = {key: value.detach().cpu() for key, value in train_dataset.items()}
    validation_dataset_cpu = {key: value.detach().cpu() for key, value in validation_dataset.items()}

    torch.save(
        {
            "teacher": args.teacher,
            "training_path": training_path,
            "state_dict": model.state_dict(),
            "hidden": args.hidden,
            "n_blocks": args.n_blocks,
            "num_frequencies": args.num_frequencies,
            "dtype": args.dtype,
        },
        output_dir / "model.pt",
    )
    with (output_dir / "training_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    torch.save(generated_cpu, output_dir / "generated_samples.pt")
    torch.save(validation_initial.detach().cpu(), output_dir / "target_samples.pt")
    torch.save(train_dataset_cpu, output_dir / "teacher_dataset.pt")
    torch.save(validation_dataset_cpu, output_dir / "validation_dataset.pt")

    mmd_value = s2_rbf_mmd(generated_cpu, validation_initial.detach().cpu(), seed=args.seed)
    geodesic = nearest_neighbor_geodesic_summary(generated_cpu, validation_initial.detach().cpu(), seed=args.seed)
    norm_error = generated_norm_error(generated_cpu)
    sample_finite_rate = finite_rate(generated_cpu)

    metrics = {
        "teacher": args.teacher,
        "training_path": training_path,
        "initial_train_loss": initial_train_loss,
        "final_train_loss": final_train_loss,
        "best_train_loss": best_train_loss,
        "validation_loss": validation_loss,
        "teacher_generation_seconds": teacher_generation_seconds,
        "training_seconds": training_seconds,
        "reverse_sampling_seconds": reverse_sampling_seconds,
        "s2_rbf_mmd": mmd_value,
        "nearest_neighbor_geodesic_mean": geodesic["mean"],
        "nearest_neighbor_geodesic_median": geodesic["median"],
        "nearest_neighbor_geodesic_max": geodesic["max"],
        "generated_sample_norm_error": norm_error,
        "nan_rate": 1.0 - sample_finite_rate,
        "generated_all_finite": bool(sample_finite_rate == 1.0),
        "generated_sample_count": int(generated_cpu.shape[0]),
        "device": device,
        "dtype": args.dtype,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    log(f"teacher={args.teacher} training_path={training_path}")
    log(f"final_train_loss={final_train_loss:.6e} validation_loss={validation_loss:.6e}")
    log(f"mmd={mmd_value:.6e} geodesic_mean={geodesic['mean']:.6e} norm_error={norm_error:.6e}")

    if not args.skip_viz:
        from scoremodel_ext.manifold.earthquake_smoke_viz import generate_earthquake_smoke_plots

        viz_dir = args.viz_output_dir.resolve() if args.viz_output_dir is not None else output_dir
        teacher_generated: Dict[str, torch.Tensor] = {args.teacher: generated_cpu}
        teacher_history: Dict[str, Dict[str, list[float]]] = {
            args.teacher: {
                "epochs": [int(x) for x in history.get("epochs", [])],
                "train_loss": [float(x) for x in history.get("train_loss", [])],
            }
        }

        for other in ("heat", "varadhan", "malliavin"):
            if other == args.teacher:
                continue
            other_run = output_dir.parent / other
            other_generated_path = other_run / "generated_samples.pt"
            other_history_path = other_run / "training_history.json"
            if other_generated_path.exists() and other_history_path.exists():
                teacher_generated[other] = torch.load(other_generated_path, map_location="cpu")
                with other_history_path.open("r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                teacher_history[other] = {
                    "epochs": [int(x) for x in loaded.get("epochs", [])],
                    "train_loss": [float(x) for x in loaded.get("train_loss", [])],
                }

        generate_earthquake_smoke_plots(
            observed_points=torch.cat((train_initial, validation_initial), dim=0).detach().cpu(),
            observed_train_points=train_initial.detach().cpu(),
            observed_test_points=validation_initial.detach().cpu(),
            generated_by_teacher=teacher_generated,
            training_history_by_teacher=teacher_history,
            output_dir=viz_dir,
        )


if __name__ == "__main__":
    main()
