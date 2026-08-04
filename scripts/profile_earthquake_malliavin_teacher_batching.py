#!/usr/bin/env python3
"""Profile scalar and sample-batched S2 Malliavin teacher implementations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import experiment_earthquake_teacher_compare_smoke as runner


def _execute(
    implementation: str,
    *,
    initial_points: torch.Tensor,
    times: torch.Tensor,
    noises: torch.Tensor,
    batch_size: int,
    covariance_regularization: float,
) -> dict[str, object]:
    if implementation == "scalar":
        runner.build_teacher_dataset(
            initial_points=initial_points,
            times=times,
            noises=noises,
            teacher="malliavin",
            covariance_regularization=covariance_regularization,
            heat_terms=80,
        )
        return {"effective_batch_size": 1, "oom_fallback": False}
    _, _, effective_batch_sizes = runner.build_malliavin_teacher_dataset_batched(
        initial_points=initial_points,
        times=times,
        noises=noises,
        batch_size=batch_size,
        covariance_regularization=covariance_regularization,
    )

    cursor = 0
    fallback_sizes: list[int] = []
    for effective_size in effective_batch_sizes:
        expected_size = min(batch_size, initial_points.shape[0] - cursor)
        if effective_size < expected_size:
            fallback_sizes.append(effective_size)
        cursor += effective_size
    return {
        "effective_batch_size": (
            min(fallback_sizes)
            if fallback_sizes
            else min(batch_size, initial_points.shape[0])
        ),
        "oom_fallback": bool(fallback_sizes),
    }


def measure_one(
    implementation: str,
    *,
    initial_points: torch.Tensor,
    times: torch.Tensor,
    noises: torch.Tensor,
    batch_size: int,
    covariance_regularization: float,
    repeats: int,
) -> dict:
    use_cuda = initial_points.device.type == "cuda"
    wall_times: list[float] = []
    cuda_event_times: list[float] = []
    peak_allocated_values: list[int] = []
    peak_reserved_values: list[int] = []
    effective_batch_sizes: list[int] = []
    oom_fallback = False

    for _ in range(repeats):
        if use_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_end = torch.cuda.Event(enable_timing=True)
        else:
            cuda_start = None
            cuda_end = None

        wall_start = time.perf_counter()
        if cuda_start is not None:
            cuda_start.record()
        execution_metadata = _execute(
            implementation,
            initial_points=initial_points,
            times=times,
            noises=noises,
            batch_size=batch_size,
            covariance_regularization=covariance_regularization,
        )
        if cuda_end is not None:
            cuda_end.record()
        if use_cuda:
            torch.cuda.synchronize()
        wall_times.append(time.perf_counter() - wall_start)

        if use_cuda:
            assert cuda_start is not None and cuda_end is not None
            cuda_event_times.append(cuda_start.elapsed_time(cuda_end) / 1_000.0)
            peak_allocated_values.append(torch.cuda.max_memory_allocated())
            peak_reserved_values.append(torch.cuda.max_memory_reserved())
        effective_batch_sizes.append(int(execution_metadata["effective_batch_size"]))
        oom_fallback = oom_fallback or bool(execution_metadata["oom_fallback"])

    return {
        "implementation": implementation,
        "batch_size": 1 if implementation == "scalar" else batch_size,
        "effective_batch_size": min(effective_batch_sizes),
        "oom_fallback": oom_fallback,
        "n_samples": int(initial_points.shape[0]),
        "n_steps": int(noises.shape[1]),
        "repeats": repeats,
        "wall_time_seconds": sum(wall_times) / len(wall_times),
        "cuda_event_seconds": (
            sum(cuda_event_times) / len(cuda_event_times)
            if cuda_event_times
            else None
        ),
        "peak_memory_allocated": (
            max(peak_allocated_values) if peak_allocated_values else None
        ),
        "peak_memory_reserved": (
            max(peak_reserved_values) if peak_reserved_values else None
        ),
    }


def export_trace(
    implementation: str,
    *,
    initial_points: torch.Tensor,
    times: torch.Tensor,
    noises: torch.Tensor,
    batch_size: int,
    covariance_regularization: float,
    trace_path: Path,
) -> None:
    use_cuda = initial_points.device.type == "cuda"
    activities = [torch.profiler.ProfilerActivity.CPU]
    if use_cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        _execute(
            implementation,
            initial_points=initial_points,
            times=times,
            noises=noises,
            batch_size=batch_size,
            covariance_regularization=covariance_regularization,
        )
        if use_cuda:
            torch.cuda.synchronize()

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(trace_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-initial-points-path", type=Path, required=True)
    parser.add_argument("--time-samples-path", type=Path, required=True)
    parser.add_argument("--validation-time-samples-path", type=Path, required=True)
    parser.add_argument("--teacher-noises-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, required=True)
    parser.add_argument("--validation-size", type=int, required=True)
    parser.add_argument("--n-steps", type=int, required=True)
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", choices=(1, 4, 8, 16),
        default=(1, 4, 8, 16),
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--covariance-regularization", type=float, default=1e-6)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--export-traces",
        action="store_true",
        help="Export Chrome traces in a separate profiler pass (disabled by default).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_samples < 1 or args.repeats < 1:
        raise ValueError("n-samples and repeats must be positive")
    device = runner.resolve_device(args.device)
    dtype = runner.to_dtype(args.dtype)
    initial_points, times, noises = runner.load_saved_teacher_shard_inputs(
        initial_points_path=args.teacher_initial_points_path,
        train_times_path=args.time_samples_path,
        validation_times_path=args.validation_time_samples_path,
        noises_path=args.teacher_noises_path,
        train_size=args.train_size,
        validation_size=args.validation_size,
        n_steps=args.n_steps,
        dtype=dtype,
        device=device,
    )
    n_samples = min(args.n_samples, initial_points.shape[0])
    initial_points = initial_points[:n_samples]
    times = times[:n_samples]
    noises = noises[:n_samples]
    output_dir = args.output_dir.resolve()
    results = [
        measure_one(
            "scalar",
            initial_points=initial_points,
            times=times,
            noises=noises,
            batch_size=1,
            covariance_regularization=args.covariance_regularization,
            repeats=args.repeats,
        )
    ]
    for batch_size in args.batch_sizes:
        results.append(
            measure_one(
                "batched",
                initial_points=initial_points,
                times=times,
                noises=noises,
                batch_size=batch_size,
                covariance_regularization=args.covariance_regularization,
                repeats=args.repeats,
            )
        )

    scalar_wall_time = float(results[0]["wall_time_seconds"])
    for result in results:
        result["speedup_vs_scalar"] = (
            scalar_wall_time / float(result["wall_time_seconds"])
        )

    if args.export_traces:
        for result in results:
            implementation = str(result["implementation"])
            batch_size = int(result["batch_size"])
            label = implementation if implementation == "scalar" else f"batch_{batch_size}"
            trace_path = output_dir / f"{label}_trace.json"
            export_trace(
                implementation,
                initial_points=initial_points,
                times=times,
                noises=noises,
                batch_size=batch_size,
                covariance_regularization=args.covariance_regularization,
                trace_path=trace_path,
            )
            result["trace_path"] = str(trace_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "device": str(device),
        "dtype": args.dtype,
        "export_traces": args.export_traces,
        "results": results,
    }
    with (output_dir / "teacher_batch_profile.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
