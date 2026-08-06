#!/usr/bin/env python3
"""Sample an upstream Earthquake Heat checkpoint with scoremodel_ext.

This is an inference-only bridge: it restores the upstream Haiku EMA
parameters, evaluates the raw upstream network, and performs every reverse
update with scoremodel_ext's S2 upstream-compatible GRW implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
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


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = REPOSITORY_ROOT / "upstream" / "riemannian-score-sde"
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
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="resolved upstream Hydra config (default: RUN_DIR/.hydra/config.yaml)",
    )
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


def _resolve_config_path(checkpoint_dir: Path, requested: Path | None) -> Path:
    if requested is not None:
        return _require_file(requested, "resolved upstream config")
    candidates = (
        checkpoint_dir.parent / ".hydra" / "config.yaml",
        checkpoint_dir.parent / "experiment_config.yaml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    rendered = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "the checkpoint needs its resolved Hydra config to reconstruct the "
        f"network; checked: {rendered}"
    )


def _import_upstream_dependencies() -> dict[str, Any]:
    """Import upstream-only dependencies without changing its source tree."""

    upstream_string = str(UPSTREAM_ROOT)
    if upstream_string not in sys.path:
        sys.path.insert(0, upstream_string)
    try:
        import haiku as hk
        import jax
        import jax.numpy as jnp
        from hydra.utils import get_class, instantiate
        from omegaconf import OmegaConf
        from score_sde.utils import restore
    except ImportError as error:
        raise RuntimeError(
            "upstream inference dependencies are unavailable; install the "
            "requirements from upstream/riemannian-score-sde/requirements.txt "
            "in the same environment as scoremodel_ext"
        ) from error
    return {
        "hk": hk,
        "jax": jax,
        "jnp": jnp,
        "get_class": get_class,
        "instantiate": instantiate,
        "OmegaConf": OmegaConf,
        "restore": restore,
    }


def _validate_checkpoint_config(cfg: Any) -> None:
    teacher = cfg.get("teacher")
    if teacher != "heat":
        raise ValueError(f"expected an upstream Heat checkpoint, config has {teacher=}")
    beta = cfg.get("beta_schedule")
    if beta is None:
        raise ValueError("resolved config has no beta_schedule")
    actual = (float(beta.beta_0), float(beta.beta_f), float(beta.t0), float(beta.tf))
    expected = (BETA_0, BETA_F, 0.0, TERMINAL_TIME)
    if actual != expected:
        raise ValueError(
            f"checkpoint beta schedule {actual} does not match required {expected}"
        )


def _build_raw_ema_network(
    checkpoint_dir: Path,
    config_path: Path,
) -> tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], str, dict[str, Any]]:
    dependencies = _import_upstream_dependencies()
    hk = dependencies["hk"]
    jax = dependencies["jax"]
    jnp = dependencies["jnp"]
    get_class = dependencies["get_class"]
    instantiate = dependencies["instantiate"]
    OmegaConf = dependencies["OmegaConf"]
    restore = dependencies["restore"]

    cfg = OmegaConf.load(config_path)
    _validate_checkpoint_config(cfg)
    dtype_name = str(cfg.get("dtype", "float64"))
    if dtype_name not in {"float32", "float64"}:
        raise ValueError(f"unsupported upstream dtype: {dtype_name}")
    if dtype_name == "float64":
        jax.config.update("jax_enable_x64", True)

    manifold = instantiate(cfg.manifold)

    def model(y: Any, t: Any, context: Any = None) -> Any:
        output_shape = get_class(cfg.generator._target_).output_shape(manifold)
        network = instantiate(
            cfg.generator,
            cfg.architecture,
            cfg.embedding,
            output_shape,
            manifold=manifold,
        )
        if context is not None:
            t_expanded = jnp.expand_dims(t.reshape(-1), -1)
            if context.shape[0] != y.shape[0]:
                context = jnp.repeat(jnp.expand_dims(context, 0), y.shape[0], 0)
            network_context = jnp.concatenate([t_expanded, context], axis=-1)
        else:
            network_context = t
        return network(y, network_context)

    transformed = hk.transform_with_state(model)
    train_state = restore(str(checkpoint_dir))
    params_ema = getattr(train_state, "params_ema", None)
    model_state = getattr(train_state, "model_state", None)
    if params_ema is None:
        raise ValueError("checkpoint has no EMA parameters (params_ema)")

    @jax.jit
    def apply_ema(points: Any, times: Any) -> Any:
        output, _ = transformed.apply(
            params_ema,
            model_state,
            None,
            y=points,
            t=times,
            context=None,
        )
        return output

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

    checkpoint_metadata = {
        "checkpoint_step": int(train_state.step),
        "ema_rate": float(train_state.ema_rate),
        "parameter_source": "params_ema",
        "config_teacher": str(cfg.teacher),
        "config_dtype": dtype_name,
    }
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
    config_path = _resolve_config_path(checkpoint_dir, args.config_path)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    native = _load_native_samples(native_path)
    sample_count = native.shape[0] if args.sample_count is None else args.sample_count
    raw_network_output, dtype_name, checkpoint_metadata = _build_raw_ema_network(
        checkpoint_dir,
        config_path,
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
        "config_path": str(config_path),
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
