#!/usr/bin/env python3
"""Sample an upstream Earthquake Heat checkpoint with scoremodel_ext.

This is an inference-only bridge: it restores the upstream Haiku EMA
parameters, evaluates the raw upstream network, and performs every reverse
update with scoremodel_ext's S2 upstream-compatible GRW implementation.
"""

from __future__ import annotations

import argparse
from collections import namedtuple
import json
from pathlib import Path
import pickle
from timeit import default_timer as timer
from typing import Any, Callable

import numpy as np
import torch

from scoremodel_ext.manifold.beta_schedule import LinearBetaSchedule
from scoremodel_ext.manifold.earthquake_adapter import (
    nearest_neighbor_geodesic_summary,
)
from scoremodel_ext.manifold.s2_reverse_diagnostics import (
    S2ReverseSamplerDiagnostics,
    s2_reverse_grw_upstream_style,
)


DEFAULT_RUN_DIR = Path("results/earthquake_teacher_comparison/upstream_heat")
DEFAULT_CHECKPOINT_DIR = DEFAULT_RUN_DIR / "ckpt"
DEFAULT_NATIVE_SAMPLES = DEFAULT_RUN_DIR / "generated_samples.npy"
DEFAULT_OUTPUT_DIR = Path(
    "results/earthquake_teacher_comparison/upstream_heat_ext_sampler"
)

TERMINAL_TIME = 1.0
EPSILON = 0.001
REVERSE_STEPS = 100
BETA_0 = 0.001
BETA_F = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--native-samples", type=Path, default=DEFAULT_NATIVE_SAMPLES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sample-count",
        type=int,
        default=None,
        help="number of ext samples (default: native generated_samples.npy count)",
    )
    parser.add_argument(
        "--evaluation-seed",
        type=int,
        default=0,
        help="subsampling seed for distributional comparison metrics",
    )
    args = parser.parse_args()
    if args.sample_count is not None and args.sample_count < 1:
        parser.error("--sample-count must be positive")
    return args


def _require_file(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {description}: {resolved}")
    return resolved


_TrainState = namedtuple(
    "TrainState",
    ("opt_state", "model_state", "step", "params", "ema_rate", "params_ema", "rng"),
)
_ScaleByAdamState = namedtuple("ScaleByAdamState", ("count", "mu", "nu"))
_ScaleByScheduleState = namedtuple("ScaleByScheduleState", ("count",))
_EmptyState = namedtuple("EmptyState", ())


class _CheckpointUnpickler(pickle.Unpickler):
    """Read the tree skeleton without importing legacy score_sde or Optax."""

    _COMPATIBLE_TYPES = {
        ("score_sde.utils.training", "TrainState"): _TrainState,
        ("optax._src.transform", "ScaleByAdamState"): _ScaleByAdamState,
        ("optax._src.transform", "ScaleByScheduleState"): _ScaleByScheduleState,
        ("optax._src.base", "EmptyState"): _EmptyState,
    }

    def find_class(self, module: str, name: str) -> type:
        compatible = self._COMPATIBLE_TYPES.get((module, name))
        if compatible is None:
            raise pickle.UnpicklingError(
                f"unsupported global in checkpoint tree: {module}.{name}"
            )
        return compatible


def _load_array_stream(path: Path) -> list[np.ndarray]:
    arrays = []
    stream_size = path.stat().st_size
    with path.open("rb") as handle:
        while handle.tell() < stream_size:
            arrays.append(np.load(handle, allow_pickle=False))
    if not arrays:
        raise ValueError(f"checkpoint array stream is empty: {path}")
    return arrays


def _restore_checkpoint_tree(template: Any, arrays: list[np.ndarray]) -> Any:
    """Reproduce JAX's legacy tree traversal using only Python containers."""

    position = 0

    def restore(node: Any) -> Any:
        nonlocal position
        if node is None:
            return None
        if isinstance(node, dict):
            return {key: restore(node[key]) for key in sorted(node)}
        if isinstance(node, list):
            return [restore(value) for value in node]
        if isinstance(node, tuple):
            values = [restore(value) for value in node]
            if hasattr(node, "_fields"):
                return type(node)(*values)
            return tuple(values)
        if position >= len(arrays):
            raise ValueError("checkpoint tree has more leaves than arrays.npy")
        value = arrays[position]
        position += 1
        return value

    restored = restore(template)
    if position != len(arrays):
        raise ValueError(
            "checkpoint arrays.npy has more leaves than tree.pkl: "
            f"used {position} of {len(arrays)}"
        )
    return restored


def _load_ema_checkpoint(checkpoint_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    tree_path = _require_file(checkpoint_dir / "tree.pkl", "checkpoint tree")
    arrays_path = _require_file(checkpoint_dir / "arrays.npy", "checkpoint arrays")
    with tree_path.open("rb") as handle:
        template = _CheckpointUnpickler(handle).load()
    arrays = _load_array_stream(arrays_path)
    train_state = _restore_checkpoint_tree(template, arrays)
    if not isinstance(train_state, _TrainState):
        raise TypeError("checkpoint root is not the expected TrainState")
    if not isinstance(train_state.params_ema, dict) or not train_state.params_ema:
        raise ValueError("checkpoint has no EMA parameters (params_ema)")
    metadata = {
        "checkpoint_step": int(np.asarray(train_state.step)),
        "ema_rate": float(np.asarray(train_state.ema_rate)),
        "parameter_source": "params_ema",
        "checkpoint_array_count": len(arrays),
    }
    return train_state.params_ema, metadata


def _build_raw_ema_network(
    checkpoint_dir: Path,
) -> tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], str, dict[str, Any]]:
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as error:
        raise RuntimeError("JAX must be installed in the scoremodel environment") from error

    params_ema, checkpoint_metadata = _load_ema_checkpoint(checkpoint_dir)
    parameter_arrays = [
        np.asarray(value)
        for module in params_ema.values()
        for value in module.values()
    ]
    dtype = np.result_type(*[value.dtype for value in parameter_arrays])
    if dtype not in {np.dtype("float32"), np.dtype("float64")}:
        raise ValueError(f"unsupported checkpoint parameter dtype: {dtype}")
    dtype_name = dtype.name
    jax.config.update("jax_enable_x64", dtype_name == "float64")

    hidden_shapes = []
    module_names = sorted(params_ema)
    expected_names = [
        "div_free_generator/~/concat/linear",
        *[
            f"div_free_generator/~/concat/linear_{index}"
            for index in range(1, 6)
        ],
    ]
    if module_names != expected_names:
        raise ValueError(
            "checkpoint is not the expected upstream DivFreeGenerator/Concat "
            f"architecture: modules={module_names}"
        )
    for name in expected_names[:-1]:
        hidden_shapes.append(int(np.asarray(params_ema[name]["b"]).shape[0]))
    if tuple(np.asarray(params_ema[expected_names[0]]["w"]).shape[:1]) != (4,):
        raise ValueError("Concat input must be three S2 coordinates plus time")
    if tuple(np.asarray(params_ema[expected_names[-1]]["b"]).shape) != (3,):
        raise ValueError("DivFreeGenerator output must have three SO(3) weights")

    layers = tuple(
        (
            jnp.asarray(params_ema[name]["w"]),
            jnp.asarray(params_ema[name]["b"]),
        )
        for name in expected_names
    )

    @jax.jit
    def apply_ema(points: Any, times: Any) -> Any:
        # Exact inference path of upstream Concat: five sin-activated hidden
        # affine layers followed by one linear three-weight output layer.
        if times.ndim == points.ndim - 1:
            times = jnp.expand_dims(times, axis=-1)
        values = jnp.concatenate([points, times], axis=-1)
        for weight, bias in layers[:-1]:
            values = jnp.sin(values @ weight + bias)
        weight, bias = layers[-1]
        weights = values @ weight + bias

        # Exact S2 DivFreeGenerator basis from the upstream geomstats fork.
        zeros = jnp.zeros_like(points[..., 0])
        f01 = jnp.stack([-points[..., 1], points[..., 0], zeros], axis=-1)
        f02 = jnp.stack([-points[..., 2], zeros, points[..., 0]], axis=-1)
        f12 = jnp.stack([zeros, -points[..., 2], points[..., 1]], axis=-1)
        generators = jnp.stack([f01, f02, f12], axis=-1)
        output = jnp.einsum("...n,...dn->...d", weights, generators)
        return output - jnp.sum(output * points, axis=-1, keepdims=True) * points

    numpy_dtype = np.float64 if dtype_name == "float64" else np.float32

    def raw_network_output(times: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        # This bridge deliberately returns the unscaled network output.  The
        # scoremodel_ext sampler applies upstream's sqrt(1-exp(-tau(t))) scale.
        points_numpy = np.asarray(points.detach().cpu().tolist(), dtype=numpy_dtype)
        times_numpy = np.asarray(times.detach().cpu().tolist(), dtype=numpy_dtype)
        output_numpy = np.asarray(apply_ema(points_numpy, times_numpy))
        # ``tolist`` also works in environments where an older Torch wheel
        # cannot use NumPy's zero-copy array bridge.
        return torch.tensor(
            output_numpy.tolist(), dtype=points.dtype, device=points.device
        )

    checkpoint_metadata["network_architecture"] = "upstream DivFreeGenerator/Concat"
    checkpoint_metadata["hidden_shapes"] = hidden_shapes
    checkpoint_metadata["config_dtype"] = dtype_name
    return raw_network_output, dtype_name, checkpoint_metadata


def _uniform_s2_and_noise(
    sample_count: int,
    *,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    terminal = torch.randn(sample_count, 3, dtype=dtype, generator=generator)
    terminal = terminal / torch.linalg.vector_norm(terminal, dim=1, keepdim=True)
    noise = torch.randn(
        REVERSE_STEPS,
        sample_count,
        3,
        dtype=dtype,
        generator=generator,
    )
    return terminal, noise


def _load_native_samples(path: Path) -> torch.Tensor:
    value = np.load(path, allow_pickle=False)
    if not isinstance(value, np.ndarray) or value.ndim != 2 or value.shape[1] != 3:
        shape = getattr(value, "shape", None)
        raise ValueError(f"native samples must have shape [sample, 3], got {shape}")
    if value.shape[0] < 1 or not bool(np.isfinite(value).all()):
        raise ValueError("native samples must be non-empty and finite")
    # Avoid depending on Torch's optional NumPy ABI bridge.  Upstream's
    # Earthquake coordinates are antipodal to scoremodel_ext's convention.
    points = -torch.tensor(value.tolist(), dtype=torch.float64)
    norms = torch.linalg.vector_norm(points, dim=1, keepdim=True)
    if bool((norms <= 0).any()) or bool(torch.max(torch.abs(norms - 1.0)) > 1e-4):
        raise ValueError("native samples are not on the unit sphere")
    return points / norms


def _summary(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().to(dtype=torch.float64, device="cpu")
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _trajectory_diagnostics(trace: S2ReverseSamplerDiagnostics) -> list[dict[str, Any]]:
    rows = []
    for step in range(REVERSE_STEPS):
        rows.append(
            {
                "step": step,
                "time": float(trace.time_grid[step]),
                "network_output_norm": _summary(trace.network_output_norm[step]),
                "score_norm": _summary(trace.score_norm[step]),
                "projected_score_norm": _summary(trace.projected_score_norm[step]),
                "drift_increment_norm": _summary(trace.drift_increment_norm[step]),
                "noise_increment_norm": _summary(trace.noise_increment_norm[step]),
                "score_standard_deviation": _summary(trace.score_std[step]),
            }
        )
    return rows


def _sample_comparison(
    generated: torch.Tensor,
    native: torch.Tensor,
    *,
    seed: int,
) -> dict[str, Any]:
    generated = generated.to(dtype=torch.float64, device="cpu")
    native = native.to(dtype=torch.float64, device="cpu")
    ext_to_native = nearest_neighbor_geodesic_summary(
        generated, native, seed=seed
    )
    native_to_ext = nearest_neighbor_geodesic_summary(
        native, generated, seed=seed
    )
    generated_mean = generated.mean(dim=0)
    native_mean = native.mean(dim=0)
    return {
        "s2_rbf_mmd": _s2_rbf_mmd(generated, native, seed=seed),
        "ext_to_native_nearest_neighbor_geodesic": ext_to_native,
        "native_to_ext_nearest_neighbor_geodesic": native_to_ext,
        "ambient_mean_vector_l2_difference": float(
            torch.linalg.vector_norm(generated_mean - native_mean)
        ),
        "ext_resultant_length": float(torch.linalg.vector_norm(generated_mean)),
        "native_resultant_length": float(torch.linalg.vector_norm(native_mean)),
    }


def _s2_rbf_mmd(
    samples: torch.Tensor,
    reference: torch.Tensor,
    *,
    sigma: float = 1.0,
    n_sub: int = 2000,
    seed: int,
) -> float:
    """Unbiased ambient RBF MMD, computed without a Torch/NumPy bridge."""

    sample_generator = torch.Generator(device="cpu").manual_seed(seed)
    reference_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    sample_index = torch.randperm(
        samples.shape[0], generator=sample_generator
    )[: min(n_sub, samples.shape[0])]
    reference_index = torch.randperm(
        reference.shape[0], generator=reference_generator
    )[: min(n_sub, reference.shape[0])]
    left = samples[sample_index]
    right = reference[reference_index]

    def gram(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        distance_sq = torch.sum(
            (first[:, None, :] - second[None, :, :]) ** 2, dim=-1
        )
        return torch.exp(-distance_sq / (2.0 * sigma**2))

    k_xx = gram(left, left)
    k_yy = gram(right, right)
    k_xy = gram(left, right)
    n_x = left.shape[0]
    n_y = right.shape[0]
    value = (
        (k_xx.sum() - torch.trace(k_xx)) / (n_x * max(n_x - 1, 1))
        + (k_yy.sum() - torch.trace(k_yy)) / (n_y * max(n_y - 1, 1))
        - 2.0 * k_xy.mean()
    )
    return float(value)


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    _require_file(checkpoint_dir / "tree.pkl", "checkpoint tree")
    _require_file(checkpoint_dir / "arrays.npy", "checkpoint arrays")
    native_path = _require_file(args.native_samples, "native generated samples")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    native = _load_native_samples(native_path)
    sample_count = native.shape[0] if args.sample_count is None else args.sample_count
    raw_network_output, dtype_name, checkpoint_metadata = _build_raw_ema_network(
        checkpoint_dir
    )
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    terminal, standard_noise = _uniform_s2_and_noise(
        int(sample_count), dtype=dtype, seed=args.seed
    )
    schedule = LinearBetaSchedule(
        beta_0=BETA_0,
        beta_f=BETA_F,
        t0=0.0,
        tf=TERMINAL_TIME,
    )

    started = timer()
    with torch.no_grad():
        trace = s2_reverse_grw_upstream_style(
            terminal,
            raw_network_output,
            standard_noise=standard_noise,
            beta_schedule=schedule,
            terminal_time=TERMINAL_TIME,
            epsilon=EPSILON,
            n_steps=REVERSE_STEPS,
            divide_network_output_by_std=True,
        )
    elapsed = timer() - started

    # Upstream's Earthquake embedding is antipodal to scoremodel_ext's standard
    # Earth convention.  Negation commutes with the S2 geometry, so convert the
    # complete artifact (not only the displayed final samples).
    generated = -trace.final_samples.detach().cpu()
    trajectory_artifact = trace.as_artifact()
    trajectory_artifact["trajectory"].neg_()
    trajectory_artifact["final_samples"].neg_()

    torch.save(generated, output_dir / "generated_samples.pt")
    torch.save(trajectory_artifact, output_dir / "reverse_trajectory.pt")

    norm_error = torch.abs(torch.linalg.vector_norm(generated, dim=1) - 1.0)
    diagnostics = {
        "inference_only": True,
        "checkpoint_dir": str(checkpoint_dir),
        "native_samples_path": str(native_path),
        "output_coordinate_system": "standard-earth",
        **checkpoint_metadata,
        "sample_count": int(sample_count),
        "native_sample_count": int(native.shape[0]),
        "seed": args.seed,
        "reverse_sampling_seconds": elapsed,
        "reverse_conditions": {
            "terminal_time": TERMINAL_TIME,
            "epsilon": EPSILON,
            "steps": REVERSE_STEPS,
            "predictor": "GRW",
            "corrector": None,
            "beta_schedule": {
                "type": "linear",
                "beta_0": BETA_0,
                "beta_f": BETA_F,
                "t0": 0.0,
                "tf": TERMINAL_TIME,
            },
            "score": "network_output / sqrt(1-exp(-tau(t)))",
            "noise": "ambient Gaussian projected to the S2 tangent space",
            "update_map": "S2 exponential map",
        },
        "maximum_unit_sphere_norm_error": float(norm_error.max()),
        "native_comparison": _sample_comparison(
            generated,
            native,
            seed=args.evaluation_seed,
        ),
        "reverse_steps": _trajectory_diagnostics(trace),
    }
    with (output_dir / "diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)

    print(f"saved {output_dir / 'generated_samples.pt'}")
    print(f"saved {output_dir / 'reverse_trajectory.pt'}")
    print(f"saved {output_dir / 'diagnostics.json'}")


if __name__ == "__main__":
    main()
