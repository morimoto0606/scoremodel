#!/usr/bin/env python3
"""Generate S^2 samples from a trained Riemannian Score-SDE checkpoint.

This script restores a saved checkpoint, builds the same sampler path used in
training/evaluation, generates samples, and writes:
- generated_samples.npy
- generated_samples.png

It intentionally avoids likelihood/log_prob and ODE likelihood evaluation.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from hydra.utils import get_class, instantiate
from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    default_ckpt = (
        Path.home()
        / "github/scoremodel/upstream/riemannian-score-sde/results/s2_toy/"
        "batch_size=32,eval_batch_size=32,steps=5000,train_plot=false,warmup_steps=100/0/ckpt"
    )
    parser = argparse.ArgumentParser(description="Plot generated S^2 samples from checkpoint")
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=default_ckpt,
        help="Path to checkpoint directory containing arrays.npy and tree.pkl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory where generated_samples.npy/png are saved",
    )
    parser.add_argument("--num-samples", type=int, default=4096, help="Number of samples")
    parser.add_argument("--sde-steps", type=int, default=100, help="PC sampler steps")
    parser.add_argument("--predictor", type=str, default="GRW", help="Predictor name")
    parser.add_argument("--seed", type=int, default=0, help="Sampling random seed")
    return parser.parse_args()


def build_model(cfg, model_manifold):
    def model_fn(y, t, context=None):
        output_shape = get_class(cfg.generator._target_).output_shape(model_manifold)
        score = instantiate(
            cfg.generator,
            cfg.architecture,
            cfg.embedding,
            output_shape,
            manifold=model_manifold,
        )
        if context is not None:
            t_expanded = jnp.expand_dims(t.reshape(-1), -1)
            if context.shape[0] != y.shape[0]:
                context = jnp.repeat(jnp.expand_dims(context, 0), y.shape[0], 0)
            context = jnp.concatenate([t_expanded, context], axis=-1)
        else:
            context = t
        return score(y, context)

    return hk.transform_with_state(model_fn)


def save_scatter_png(samples: np.ndarray, png_path: Path) -> None:
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], s=2, alpha=0.65)
    ax.set_title("Generated Samples on $S^2$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    lim = 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    # Keep imports local to avoid accidental path pollution when not needed.
    repo_root = Path(__file__).resolve().parents[1]
    upstream_root = repo_root / "upstream" / "riemannian-score-sde"
    if str(upstream_root) not in sys.path:
        sys.path.insert(0, str(upstream_root))

    from score_sde.utils import restore  # pylint: disable=import-outside-toplevel

    ckpt_dir = args.ckpt_dir.expanduser().resolve()
    run_dir = ckpt_dir.parent
    cfg_path = run_dir / ".hydra" / "config.yaml"

    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    if not (ckpt_dir / "arrays.npy").exists() or not (ckpt_dir / "tree.pkl").exists():
        raise FileNotFoundError(f"Checkpoint files missing in: {ckpt_dir}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Hydra config not found: {cfg_path}")

    os.environ.setdefault("JAX_ENABLE_X64", "True")
    os.environ.setdefault("GEOMSTATS_BACKEND", "jax")

    cfg = OmegaConf.load(cfg_path)

    data_manifold = instantiate(cfg.manifold)
    transform = instantiate(cfg.transform, data_manifold)
    model_manifold = transform.domain
    beta_schedule = instantiate(cfg.beta_schedule)
    flow = instantiate(cfg.flow, manifold=model_manifold, beta_schedule=beta_schedule)
    base = instantiate(cfg.base, model_manifold, flow)
    pushforward = instantiate(cfg.pushf, flow, base, transform=transform)

    model = build_model(cfg, model_manifold)
    train_state = restore(str(ckpt_dir))

    model_w_dicts = (model, train_state.params_ema, train_state.model_state)
    sampler = pushforward.get_sampler(
        model_w_dicts,
        train=False,
        N=args.sde_steps,
        eps=float(cfg.eps),
        predictor=args.predictor,
    )

    rng = jax.random.PRNGKey(args.seed)
    samples = sampler(rng, (args.num_samples,), context=None)
    samples_np = np.asarray(samples)

    if samples_np.ndim != 2 or samples_np.shape[1] != 3:
        raise ValueError(f"Expected sample shape (N, 3), got {samples_np.shape}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_path = output_dir / "generated_samples.npy"
    png_path = output_dir / "generated_samples.png"

    np.save(npy_path, samples_np)
    save_scatter_png(samples_np, png_path)

    print(f"checkpoint_path={ckpt_dir}")
    print(f"sample_shape={samples_np.shape}")
    print(f"figure_path={png_path}")


if __name__ == "__main__":
    main()
