#!/usr/bin/env python3
"""Generate S2_Malliavin samples under Debortoli-like conditions on GPU."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from scoremodel_ext.manifold.experiment_s2_malliavin_teacher import _build_heat_score_function
from scoremodel_ext.manifold.s2_malliavin import s2_reverse_grw


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("results/s2_malliavin_teacher_exact"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=4096)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--terminal-time", type=float, default=0.3)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--score-mode", choices=("heat", "varadhan"), default="heat")
    parser.add_argument("--heat-terms", type=int, default=80)
    parser.add_argument("--init-mode", choices=("forward", "uniform"), default="forward")
    parser.add_argument("--tag", type=str, default="same_conditions_gpu")
    return parser.parse_args()


def _uniform_s2_samples(n_samples: int, *, dtype: torch.dtype, device: str, generator: torch.Generator) -> torch.Tensor:
    samples = torch.randn(n_samples, 3, dtype=dtype, device=device, generator=generator)
    return samples / torch.linalg.vector_norm(samples, dim=1, keepdim=True)


def _build_varadhan_score_function(x0: torch.Tensor):
    def _score_fn(t_batch: torch.Tensor, x_batch: torch.Tensor) -> torch.Tensor:
        cosine = torch.clamp((x_batch * x0).sum(dim=1), -1.0, 1.0)
        angle = torch.acos(cosine)
        tangent = x0.unsqueeze(0) - cosine.unsqueeze(1) * x_batch
        tangent_norm = torch.linalg.vector_norm(tangent, dim=1)
        base_scale = torch.where(
            tangent_norm > 1e-10,
            angle / tangent_norm,
            torch.ones_like(tangent_norm),
        )
        return (base_scale / t_batch).unsqueeze(1) * tangent

    return _score_fn


def main() -> None:
    args = _parse_args()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    seed = args.seed
    n_samples = args.n_samples
    n_steps = args.n_steps
    terminal_time = args.terminal_time

    x0 = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    if args.init_mode == "forward":
        points = x0.repeat(n_samples, 1)
        sqrt_dt = (terminal_time / n_steps) ** 0.5
        # Forward GRW with projected ambient noise on S2.
        for _ in range(n_steps):
            noise = torch.randn(n_samples, 3, generator=rng, dtype=dtype, device=device)
            tangent_noise = noise - (noise * points).sum(dim=1, keepdim=True) * points
            increment = sqrt_dt * tangent_noise
            norm = torch.linalg.vector_norm(increment, dim=1, keepdim=True)
            points = torch.cos(norm) * points + torch.sinc(norm / np.pi) * increment
            points = points / torch.linalg.vector_norm(points, dim=1, keepdim=True)
        endpoints = points
    else:
        endpoints = _uniform_s2_samples(n_samples, dtype=dtype, device=device, generator=rng)

    if args.score_mode == "heat":
        score_fn = _build_heat_score_function(x0, n_heat_terms=args.heat_terms)
    else:
        score_fn = _build_varadhan_score_function(x0)

    reverse_noise = torch.randn(n_steps, n_samples, 3, generator=rng, dtype=dtype, device=device)
    generated = s2_reverse_grw(
        endpoints,
        score_fn,
        terminal_time=terminal_time,
        n_steps=n_steps,
        standard_noise=reverse_noise,
    )

    end_np = endpoints.detach().cpu().numpy()
    gen_np = generated.detach().cpu().numpy()

    np.save(outdir / f"generated_samples_{args.tag}.npy", gen_np)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(gen_np[:, 0], gen_np[:, 1], gen_np[:, 2], s=2, alpha=0.65)
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
    fig.savefig(outdir / f"generated_samples_{args.tag}.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(end_np[:, 0], end_np[:, 1], end_np[:, 2], s=2, alpha=0.65)
    ax.set_title("Forward Endpoints on $S^2$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    fig.savefig(outdir / f"forward_endpoints_{args.tag}.png", dpi=180)
    plt.close(fig)

    x0_np = x0.detach().cpu().numpy()
    end_resultant = np.linalg.norm(end_np.mean(axis=0))
    gen_resultant = np.linalg.norm(gen_np.mean(axis=0))
    end_ang = np.degrees(np.arccos(np.clip(end_np @ x0_np, -1.0, 1.0)))
    gen_ang = np.degrees(np.arccos(np.clip(gen_np @ x0_np, -1.0, 1.0)))

    print(f"device={device}")
    print("score_mode", args.score_mode)
    print("init_mode", args.init_mode)
    print("saved", outdir / f"generated_samples_{args.tag}.png")
    print("saved", outdir / f"forward_endpoints_{args.tag}.png")
    print("saved", outdir / f"generated_samples_{args.tag}.npy")
    print("forward_resultant", float(end_resultant))
    print("generated_resultant", float(gen_resultant))
    print("forward_angle_mean_deg", float(end_ang.mean()))
    print("generated_angle_mean_deg", float(gen_ang.mean()))


if __name__ == "__main__":
    main()
