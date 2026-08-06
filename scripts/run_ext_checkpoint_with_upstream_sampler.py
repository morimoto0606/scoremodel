#!/usr/bin/env python3
"""Run an ext Heat checkpoint with an upstream-compatible S2 sampler.

This script is inference-only.  It does not train, alter the saved checkpoint,
or modify either production reverse sampler.  The ext checkpoint output is an
effective score and is therefore passed to the upstream-style update without
the upstream raw-network-output standard-deviation scaling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Mapping

import torch

if __package__:
    from .experiment_earthquake_teacher_compare_smoke import (
        build_model_from_training_checkpoint,
        build_score_fn,
        load_run_config_for_model,
        require_exact_checkpoint_state,
        resolve_device,
        to_dtype,
    )
else:
    from experiment_earthquake_teacher_compare_smoke import (
        build_model_from_training_checkpoint,
        build_score_fn,
        load_run_config_for_model,
        require_exact_checkpoint_state,
        resolve_device,
        to_dtype,
    )

from scoremodel_ext.manifold.beta_schedule import LinearBetaSchedule
from scoremodel_ext.manifold.s2_malliavin import s2_reverse_grw
from scoremodel_ext.manifold.s2_reverse_diagnostics import (
    S2ReverseSamplerDiagnostics,
    s2_reverse_grw_upstream_style,
    trace_s2_reverse_grw_current_style,
)


Tensor = torch.Tensor
ScoreFunction = Callable[[Tensor, Tensor], Tensor]

DEFAULT_CHECKPOINT = Path("results/earthquake_linear_beta_100k_ema_heat/model.pt")
DEFAULT_OUTPUT_DIR = Path(
    "results/earthquake_teacher_comparison/ext_heat_upstream_sampler"
)
TERMINAL_TIME = 1.0
EPSILON = 0.001
REVERSE_STEPS = 100
BETA_0 = 0.001
BETA_F = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--terminal-samples-path", type=Path, default=None)
    parser.add_argument("--reverse-noise-path", type=Path, default=None)
    args = parser.parse_args()
    if args.sample_count is not None and args.sample_count < 1:
        parser.error("--sample-count must be positive")
    return args


def upstream_schedule() -> LinearBetaSchedule:
    return LinearBetaSchedule(
        beta_0=BETA_0,
        beta_f=BETA_F,
        t0=0.0,
        tf=TERMINAL_TIME,
    )


def _load_tensor(path: Path, *, key: str) -> Tensor:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing tensor artifact: {resolved}")
    value = torch.load(resolved, map_location="cpu")
    if isinstance(value, Mapping):
        if key not in value:
            raise KeyError(f"artifact {resolved} has no key {key!r}")
        value = value[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"artifact {resolved} does not contain a tensor")
    return value


def make_reverse_inputs(
    sample_count: int,
    *,
    dtype: torch.dtype,
    device: str | torch.device,
    seed: int,
    n_steps: int = REVERSE_STEPS,
) -> tuple[Tensor, Tensor]:
    """Create reproducible uniform-S2 endpoints and ambient Gaussian noise."""

    if sample_count < 1 or n_steps < 1:
        raise ValueError("sample_count and n_steps must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    terminal = torch.randn(sample_count, 3, dtype=dtype, generator=generator)
    terminal = terminal / torch.linalg.vector_norm(terminal, dim=1, keepdim=True)
    noise = torch.randn(n_steps, sample_count, 3, dtype=dtype, generator=generator)
    return terminal.to(device=device), noise.to(device=device)


def run_upstream_compatible_reverse(
    terminal_samples: Tensor,
    effective_score_fn: ScoreFunction,
    standard_noise: Tensor,
    *,
    beta_schedule: LinearBetaSchedule | None = None,
    terminal_time: float = TERMINAL_TIME,
    epsilon: float = EPSILON,
    n_steps: int = REVERSE_STEPS,
) -> S2ReverseSamplerDiagnostics:
    """Apply upstream reverse GRW without rescaling the effective score."""

    return s2_reverse_grw_upstream_style(
        terminal_samples,
        effective_score_fn,
        standard_noise=standard_noise,
        beta_schedule=upstream_schedule() if beta_schedule is None else beta_schedule,
        terminal_time=terminal_time,
        epsilon=epsilon,
        n_steps=n_steps,
        # Critical: ext checkpoints already return the effective score.
        divide_network_output_by_std=False,
    )


def _summary(values: Tensor) -> dict[str, float]:
    values = values.detach().to(dtype=torch.float64, device="cpu")
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _per_step_summary(values: Tensor) -> list[dict[str, float]]:
    return [_summary(row) for row in values]


def _final_comparison(left: Tensor, right: Tensor) -> dict[str, float]:
    left = left.detach().to(dtype=torch.float64, device="cpu")
    right = right.detach().to(dtype=torch.float64, device="cpu")
    difference = torch.abs(left - right)
    distance = torch.acos(torch.sum(left * right, dim=1).clamp(-1.0, 1.0))
    return {
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "mean_geodesic_distance": float(distance.mean()),
        "median_geodesic_distance": float(distance.median()),
    }


def _resolve_inputs(
    args: argparse.Namespace,
    *,
    sample_count: int,
    dtype: torch.dtype,
    device: str,
) -> tuple[Tensor, Tensor]:
    generated_terminal, generated_noise = make_reverse_inputs(
        sample_count,
        dtype=dtype,
        device=device,
        seed=args.seed,
    )
    terminal = (
        generated_terminal
        if args.terminal_samples_path is None
        else _load_tensor(args.terminal_samples_path, key="terminal_samples")
        .to(device=device, dtype=dtype)[:sample_count]
    )
    noise = (
        generated_noise
        if args.reverse_noise_path is None
        else _load_tensor(args.reverse_noise_path, key="reverse_noise")
        .to(device=device, dtype=dtype)[:, :sample_count]
    )
    expected_terminal = (sample_count, 3)
    expected_noise = (REVERSE_STEPS, sample_count, 3)
    if tuple(terminal.shape) != expected_terminal:
        raise ValueError(
            f"terminal samples must have shape {expected_terminal}, got {tuple(terminal.shape)}"
        )
    if tuple(noise.shape) != expected_noise:
        raise ValueError(
            f"reverse noise must have shape {expected_noise}, got {tuple(noise.shape)}"
        )
    return terminal, noise


def main() -> None:
    args = parse_args()
    checkpoint_path, run_config = load_run_config_for_model(args.checkpoint_path)
    if str(run_config.get("teacher")) != "heat":
        raise ValueError("the checkpoint must be an ext Heat model")
    device = resolve_device(args.device)
    dtype = to_dtype(str(run_config["dtype"]))
    sample_count = (
        int(run_config.get("n_generated_samples", 4096))
        if args.sample_count is None
        else args.sample_count
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model_from_training_checkpoint(checkpoint_path, device=device)
    require_exact_checkpoint_state(model, checkpoint_path)
    effective_score_fn = build_score_fn(model)
    terminal, noise = _resolve_inputs(
        args,
        sample_count=sample_count,
        dtype=dtype,
        device=device,
    )
    schedule = upstream_schedule()

    with torch.no_grad():
        current = trace_s2_reverse_grw_current_style(
            terminal,
            effective_score_fn,
            terminal_time=TERMINAL_TIME,
            n_steps=REVERSE_STEPS,
            standard_noise=noise,
            minimum_forward_time=EPSILON,
            beta_schedule=schedule,
        )
        production_current = s2_reverse_grw(
            terminal,
            effective_score_fn,
            terminal_time=TERMINAL_TIME,
            n_steps=REVERSE_STEPS,
            standard_noise=noise,
            minimum_forward_time=EPSILON,
            beta_schedule=schedule,
        )
        torch.testing.assert_close(
            current.final_samples,
            production_current,
            rtol=0,
            atol=1e-12,
        )
        upstream = run_upstream_compatible_reverse(
            terminal,
            effective_score_fn,
            noise,
            beta_schedule=schedule,
        )

    generated = upstream.final_samples.detach().cpu()
    current_artifact = current.as_artifact()
    upstream_artifact = upstream.as_artifact()
    torch.save(generated, output_dir / "generated_samples.pt")
    torch.save(upstream_artifact, output_dir / "reverse_trajectory.pt")

    # Aliases match compare_earthquake_reverse_samplers.py so its consumers can
    # inspect this run without learning another artifact schema.
    torch.save(current_artifact, output_dir / "current_style_reverse_trajectory.pt")
    torch.save(production_current.detach().cpu(), output_dir / "current_style_final_samples.pt")
    torch.save(upstream_artifact, output_dir / "upstream_style_reverse_trajectory.pt")
    torch.save(generated, output_dir / "upstream_style_final_samples.pt")
    torch.save(terminal.detach().cpu(), output_dir / "terminal_samples.pt")
    torch.save(noise.detach().cpu(), output_dir / "comparison_reverse_noise.pt")

    norm_error = torch.abs(torch.linalg.vector_norm(generated, dim=1) - 1.0)
    diagnostics = {
        "checkpoint_path": str(checkpoint_path),
        "sample_count": sample_count,
        "dtype": str(run_config["dtype"]),
        "device": device,
        "reverse_steps": REVERSE_STEPS,
        "terminal_time": TERMINAL_TIME,
        "epsilon": EPSILON,
        "endpoint_inclusive_time_grid": True,
        "signed_dt": (EPSILON - TERMINAL_TIME) / REVERSE_STEPS,
        "beta_schedule": {
            "type": "linear",
            "beta_0": BETA_0,
            "beta_f": BETA_F,
            "t0": 0.0,
            "tf": TERMINAL_TIME,
        },
        "score_input_type": "effective_score",
        "score_definition": "score = checkpoint_output",
        "score_standard_deviation_scaling_applied": False,
        "network_output_norm": _per_step_summary(upstream.network_output_norm),
        "score_norm": _per_step_summary(upstream.score_norm),
        "projected_score_norm": _per_step_summary(upstream.projected_score_norm),
        "beta_score_norm": _per_step_summary(upstream.beta_score_norm),
        "drift_increment_norm": _per_step_summary(upstream.drift_increment_norm),
        "noise_increment_norm": _per_step_summary(upstream.noise_increment_norm),
        "maximum_unit_sphere_norm_error": float(norm_error.max()),
        "existing_vs_upstream_compatible": _final_comparison(
            production_current,
            upstream.final_samples,
        ),
        "current_trace_matches_existing_sampler_atol": 1e-12,
        "artifact_schema": "compare_earthquake_reverse_samplers.py compatible",
    }
    with (output_dir / "diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)

    print(f"saved ext upstream-compatible artifacts in {output_dir}")


if __name__ == "__main__":
    main()
