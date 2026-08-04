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
) -> None:
    if implementation == "scalar":
        runner.build_teacher_dataset(
            initial_points=initial_points,
            times=times,
            noises=noises,
            teacher="malliavin",
            covariance_regularization=covariance_regularization,
            heat_terms=80,
        )
        return
    runner.build_malliavin_teacher_dataset_batched(
        initial_points=initial_points,
        times=times,
        noises=noises,
        batch_size=batch_size,
        covariance_regularization=covariance_regularization,
    )


def profile_one(
    implementation: str,
    *,
    initial_points: torch.Tensor,
    times: torch.Tensor,
    noises: torch.Tensor,
    batch_size: int,
    covariance_regularization: float,
    repeats: int,
    trace_path: Path,
) -> dict:
    use_cuda = initial_points.device.type == "cuda"
    activities = [torch.profiler.ProfilerActivity.CPU]
    if use_cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        cuda_start = torch.cuda.Event(enable_timing=True)
        cuda_end = torch.cuda.Event(enable_timing=True)
    else:
        cuda_start = None
        cuda_end = None

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        wall_start = time.perf_counter()
        if cuda_start is not None:
            cuda_start.record()
        for _ in range(repeats):
            _execute(
                implementation,
                initial_points=initial_points,
                times=times,
                noises=noises,
                batch_size=batch_size,
                covariance_regularization=covariance_regularization,
            )
        if cuda_end is not None:
            cuda_end.record()
            torch.cuda.synchronize()
        wall_seconds = time.perf_counter() - wall_start

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(trace_path))
    if use_cuda:
        cuda_event_ms = cuda_start.elapsed_time(cuda_end) / repeats
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        cuda_device_type = torch.autograd.DeviceType.CUDA
        kernel_count = sum(
            1
            for event in profiler.events()
            if event.device_type == cuda_device_type
        )
        kernel_count /= repeats
    else:
        cuda_event_ms = None
        peak_allocated = None
        peak_reserved = None
        kernel_count = None
    return {
        "implementation": implementation,
        "batch_size": 1 if implementation == "scalar" else batch_size,
        "n_samples": int(initial_points.shape[0]),
        "n_steps": int(noises.shape[1]),
        "repeats": repeats,
        "wall_seconds_per_repeat": wall_seconds / repeats,
        "cuda_event_ms_per_repeat": cuda_event_ms,
        "peak_memory_allocated_bytes": peak_allocated,
        "peak_memory_reserved_bytes": peak_reserved,
        "cuda_kernel_count_per_repeat": kernel_count,
        "trace_path": str(trace_path),
    }


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
        profile_one(
            "scalar",
            initial_points=initial_points,
            times=times,
            noises=noises,
            batch_size=1,
            covariance_regularization=args.covariance_regularization,
            repeats=args.repeats,
            trace_path=output_dir / "scalar_trace.json",
        )
    ]
    for batch_size in args.batch_sizes:
        results.append(
            profile_one(
                "batched",
                initial_points=initial_points,
                times=times,
                noises=noises,
                batch_size=batch_size,
                covariance_regularization=args.covariance_regularization,
                repeats=args.repeats,
                trace_path=output_dir / f"batch_{batch_size}_trace.json",
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "device": device,
        "dtype": args.dtype,
        "results": results,
    }
    with (output_dir / "teacher_batch_profile.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
