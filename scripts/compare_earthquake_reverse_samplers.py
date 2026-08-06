#!/usr/bin/env python3
"""Compare current and upstream-style Earthquake S2 reverse samplers.

Only saved checkpoints and reverse inputs are read.  Training and teacher
generation are never invoked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import torch

from experiment_earthquake_teacher_compare_smoke import (
    beta_schedule_from_run_config,
    build_model_from_training_checkpoint,
    build_score_fn,
    load_run_config_for_model,
    require_exact_checkpoint_state,
    resolve_device,
    to_dtype,
)
from scoremodel_ext.manifold.s2_malliavin import s2_reverse_grw
from scoremodel_ext.manifold.s2_reverse_diagnostics import (
    UPSTREAM_REVERSE_EPSILON,
    UPSTREAM_REVERSE_STEPS,
    S2ReverseSamplerDiagnostics,
    s2_reverse_grw_upstream_style,
    trace_s2_reverse_grw_current_style,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--terminal-samples-path", type=Path, default=None)
    parser.add_argument("--reverse-noise-path", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-points", type=int, default=4096)
    parser.add_argument("--reverse-seed", type=int, default=271828)
    parser.add_argument(
        "--upstream-style-reverse",
        action="store_true",
        help="also run the upstream N=100, epsilon=0.001 parameterisation",
    )
    parser.add_argument(
        "--upstream-score-input",
        choices=("raw-network-output", "effective-score"),
        default="raw-network-output",
        help=(
            "raw-network-output reproduces upstream's /std wrapper; "
            "effective-score uses a scoremodel_ext output that already represents score"
        ),
    )
    args = parser.parse_args()
    if args.max_points < 1:
        parser.error("--max-points must be positive")
    return args


def _load_tensor(path: Path, *, key: str) -> torch.Tensor:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing artifact: {resolved}")
    value = torch.load(resolved, map_location="cpu")
    if isinstance(value, Mapping):
        if key not in value:
            raise KeyError(f"artifact {resolved} has no key {key!r}")
        value = value[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"artifact {resolved} does not contain a tensor")
    return value


def _load_terminal_samples(path: Path, *, maximum: int) -> torch.Tensor:
    points = _load_tensor(path, key="terminal_samples")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"terminal samples must have shape [batch, 3], got {points.shape}")
    return points[:maximum]


def _load_or_create_noise(
    path: Path | None,
    *,
    output_path: Path,
    sample_count: int,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    expected_shape = (UPSTREAM_REVERSE_STEPS, sample_count, 3)
    if path is not None:
        noise = _load_tensor(path, key="reverse_noise")
        if noise.ndim != 3 or noise.shape[0] != UPSTREAM_REVERSE_STEPS:
            raise ValueError(
                "upstream comparison noise must have shape "
                f"[100, batch, 3], got {tuple(noise.shape)}"
            )
        noise = noise[:, :sample_count]
        if tuple(noise.shape) != expected_shape:
            raise ValueError(f"reverse noise must have shape {expected_shape}")
        return noise.to(dtype=dtype)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(expected_shape, generator=generator, dtype=dtype)
    torch.save(noise, output_path)
    return noise


def _summary(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().to(dtype=torch.float64, device="cpu")
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _trace_summary(trace: S2ReverseSamplerDiagnostics) -> list[dict[str, object]]:
    rows = []
    for step in range(trace.time_grid.shape[0]):
        rows.append(
            {
                "step": step,
                "time": float(trace.time_grid[step].detach().cpu()),
                "network_output_norm": _summary(trace.network_output_norm[step]),
                "score_norm": _summary(trace.score_norm[step]),
                "projected_score_norm": _summary(trace.projected_score_norm[step]),
                "beta_score_norm": _summary(trace.beta_score_norm[step]),
                "beta_projected_score_norm": _summary(
                    trace.beta_projected_score_norm[step]
                ),
                "drift_increment_norm": _summary(trace.drift_increment_norm[step]),
                "noise_increment_norm": _summary(trace.noise_increment_norm[step]),
                "drift_noise_ratio": _summary(
                    trace.drift_increment_norm[step]
                    / trace.noise_increment_norm[step].clamp_min(1e-12)
                ),
                "score_std": _summary(trace.score_std[step]),
            }
        )
    return rows


def _final_comparison(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    difference = torch.abs(left - right)
    dot = torch.sum(left * right, dim=1).clamp(-1.0, 1.0)
    distance = torch.acos(dot)
    return {
        "max_abs_error": float(difference.max().detach().cpu()),
        "mean_abs_error": float(difference.mean().detach().cpu()),
        "mean_geodesic_distance": float(distance.mean().detach().cpu()),
        "median_geodesic_distance": float(distance.median().detach().cpu()),
    }


def _save_plot(
    current: S2ReverseSamplerDiagnostics,
    upstream: S2ReverseSamplerDiagnostics | None,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=220)
    traces = [("current", current)]
    if upstream is not None:
        traces.append(("upstream-style", upstream))
    for label, trace in traces:
        times = trace.time_grid.detach().cpu()
        axes[0].plot(
            times,
            trace.score_norm.mean(dim=1).detach().cpu(),
            label=f"{label}: score",
        )
        axes[0].plot(
            times,
            trace.network_output_norm.mean(dim=1).detach().cpu(),
            linestyle=":",
            label=f"{label}: network output",
        )
        axes[1].plot(
            times,
            trace.beta_score_norm.mean(dim=1).detach().cpu(),
            label=f"{label}: beta*score",
        )
        axes[1].plot(
            times,
            trace.beta_projected_score_norm.mean(dim=1).detach().cpu(),
            linestyle="--",
            label=f"{label}: beta*projected score",
        )
        axes[1].plot(
            times,
            trace.noise_increment_norm.mean(dim=1).detach().cpu(),
            linestyle=":",
            label=f"{label}: noise increment",
        )
    axes[0].set_title("Score parameterisation")
    axes[0].set_xlabel("Physical time t")
    axes[0].set_ylabel("Mean norm")
    axes[1].set_title("Effective reverse terms")
    axes[1].set_xlabel("Physical time t")
    axes[1].set_ylabel("Mean norm")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        axis.invert_xaxis()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    model_path, run_config = load_run_config_for_model(args.model_path)
    output_dir = (
        model_path.parent / "reverse_sampler_comparison"
        if args.output_dir is None
        else args.output_dir.expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    dtype = to_dtype(str(run_config["dtype"]))
    schedule = beta_schedule_from_run_config(run_config)
    model = build_model_from_training_checkpoint(model_path, device=device)
    require_exact_checkpoint_state(model, model_path)
    score_fn = build_score_fn(model)

    terminal_path = (
        model_path.parent / "terminal_samples.pt"
        if args.terminal_samples_path is None
        else args.terminal_samples_path.expanduser().resolve()
    )
    terminal_samples = _load_terminal_samples(
        terminal_path,
        maximum=args.max_points,
    ).to(device=device, dtype=dtype)
    noise_path = (
        None
        if args.reverse_noise_path is None
        else args.reverse_noise_path.expanduser().resolve()
    )
    noise_output_path = output_dir / "comparison_reverse_noise.pt"
    reverse_noise = _load_or_create_noise(
        noise_path,
        output_path=noise_output_path,
        sample_count=terminal_samples.shape[0],
        dtype=dtype,
        seed=args.reverse_seed,
    ).to(device=device, dtype=dtype)

    with torch.no_grad():
        current = trace_s2_reverse_grw_current_style(
            terminal_samples,
            score_fn,
            terminal_time=1.0,
            n_steps=UPSTREAM_REVERSE_STEPS,
            standard_noise=reverse_noise,
            minimum_forward_time=UPSTREAM_REVERSE_EPSILON,
            beta_schedule=schedule,
        )
        production_final = s2_reverse_grw(
            terminal_samples,
            score_fn,
            terminal_time=1.0,
            n_steps=UPSTREAM_REVERSE_STEPS,
            standard_noise=reverse_noise,
            minimum_forward_time=UPSTREAM_REVERSE_EPSILON,
            beta_schedule=schedule,
        )
        torch.testing.assert_close(
            current.final_samples,
            production_final,
            rtol=0,
            atol=1e-12,
        )
        upstream = (
            s2_reverse_grw_upstream_style(
                terminal_samples,
                score_fn,
                standard_noise=reverse_noise,
                beta_schedule=schedule,
                terminal_time=1.0,
                epsilon=UPSTREAM_REVERSE_EPSILON,
                n_steps=UPSTREAM_REVERSE_STEPS,
                divide_network_output_by_std=(
                    args.upstream_score_input == "raw-network-output"
                ),
            )
            if args.upstream_style_reverse
            else None
        )

    torch.save(current.as_artifact(), output_dir / "current_style_reverse_trajectory.pt")
    torch.save(
        current.final_samples.detach().cpu(),
        output_dir / "current_style_final_samples.pt",
    )
    if upstream is not None:
        torch.save(
            upstream.as_artifact(),
            output_dir / "upstream_style_reverse_trajectory.pt",
        )
        torch.save(
            upstream.final_samples.detach().cpu(),
            output_dir / "upstream_style_final_samples.pt",
        )

    report: dict[str, object] = {
        "model_path": str(model_path),
        "terminal_samples_path": str(terminal_path),
        "reverse_noise_path": str(noise_path or noise_output_path),
        "sample_count": int(terminal_samples.shape[0]),
        "dtype": str(run_config["dtype"]),
        "device": device,
        "reverse_steps": UPSTREAM_REVERSE_STEPS,
        "terminal_time": 1.0,
        "epsilon": UPSTREAM_REVERSE_EPSILON,
        "upstream_score_input": args.upstream_score_input,
        "current_trace_matches_production_atol": 1e-12,
        "current_style": _trace_summary(current),
        "upstream_style": None,
        "final_sample_comparison": None,
    }
    if upstream is not None:
        report["upstream_style"] = _trace_summary(upstream)
        report["final_sample_comparison"] = _final_comparison(
            current.final_samples,
            upstream.final_samples,
        )
    report_path = output_dir / "reverse_sampler_comparison.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    plot_path = output_dir / "reverse_sampler_score_norm.png"
    _save_plot(current, upstream, plot_path)
    print(f"saved {report_path}")
    print(f"saved {plot_path}")


if __name__ == "__main__":
    main()
