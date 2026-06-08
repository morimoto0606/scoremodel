"""
SwissRoll linear-VP vs nonlinear comparison (linear side runner)
================================================================

This script runs the linear VP baseline with the same SwissRoll settings used
for the nonlinear Mirafzali low-time experiments, then writes per-seed metrics
and aggregate summaries.

Design principle:
- This runner isolates only the forward-SDE choice.
- It uses the closed-form linear Malliavin teacher from sde_linear.py.
- It does not use nonlinear simulators or correction variants.

For now this runner executes only the linear VP branch. Nonlinear results are
expected to already exist under:
  results/mirafzali_variance_diag_5seed/swissroll

Run
---
python -m scoremodel_ext.malliavin.experiment_linear_vs_nonlinear_swissroll
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

try:
    from .datasets_2d import get_sampler, sample_8gmm
    from .evaluation import compute_metrics_nl
    from .experiment_mirafzali import _plot_scatter, _plot_teacher_field, build_training_dataset
    from .models import train_mirafzali_skorokhod_net
    from .sde_linear import VPConfig, reverse_sample_linear, simulate_linear
except ImportError:
    from datasets_2d import get_sampler, sample_8gmm
    from evaluation import compute_metrics_nl
    from experiment_mirafzali import _plot_scatter, _plot_teacher_field, build_training_dataset
    from models import train_mirafzali_skorokhod_net
    from sde_linear import VPConfig, reverse_sample_linear, simulate_linear


TIMES = [
    0.005, 0.01, 0.02, 0.035,
    0.05, 0.075, 0.10,
    0.20, 0.35, 0.50, 0.75, 1.00,
]


def _simulate_all_times_linear_vp(
    dataset: str,
    times: Sequence[float],
    cfg: VPConfig,
    n_paths: int,
    device: str,
):
    sampler = get_sampler(dataset)
    sim_cache = []
    for T in times:
        result = sampler(n_paths, device=device)
        X0 = result[0] if isinstance(result, tuple) else result
        X_T, H = simulate_linear(X0, T, cfg)
        sim_cache.append((T, X_T, H))
    return sim_cache


def run_linear_vp_seed(
    dataset: str = "swissroll",
    seed: int = 0,
    cfg: VPConfig = VPConfig(beta_min=1.0, beta_max=25.0, T=1.0),
    times: Sequence[float] = TIMES,
    hidden: int = 2048,
    n_blocks: int = 6,
    n_paths: int = 25_000,
    n_epochs: int = 5_000,
    batch_size: int = 4_096,
    lr: float = 2e-4,
    n_samples_reverse: int = 20_000,
    n_steps_rev: int = 1_000,
    outbase: str = "results/linear_vs_nonlinear_swissroll_lowt_stationary_5seed",
    device: Optional[str] = None,
) -> dict:
    run_start = time.perf_counter()

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    outdir = Path(outbase) / "linear_vp" / f"seed{seed}"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  linear_vp  dataset={dataset}  seed={seed}  device={device}")
    print(f"{'='*60}", flush=True)

    sim_start = time.perf_counter()
    sim_cache = _simulate_all_times_linear_vp(
        dataset=dataset,
        times=times,
        cfg=cfg,
        n_paths=n_paths,
        device=device,
    )
    sim_seconds = time.perf_counter() - sim_start

    teacher_start = time.perf_counter()
    dataset_tuple = build_training_dataset(
        sim_cache,
        method="raw",
        n_raw=n_paths,
    )
    if dataset_tuple is None:
        raise RuntimeError("No valid teacher points found for linear VP.")

    t_tr, x_tr, s_tr, _ = dataset_tuple
    teacher_seconds = time.perf_counter() - teacher_start

    T_last = float(times[-1])
    t_mask = torch.isclose(t_tr, torch.full_like(t_tr, T_last), atol=1e-7)
    pts_np = x_tr[t_mask].detach().cpu().numpy()
    sc_np = s_tr[t_mask].detach().cpu().numpy()
    if len(pts_np) > 0:
        _plot_teacher_field(
            pts_np,
            sc_np,
            T_last,
            f"Teacher field (linear_vp), T={T_last:.2f}",
            outdir / "teacher_field.png",
        )

        # Main visual diagnostic: forward teacher samples at terminal time.
        _plot_scatter(
            pts_np,
            f"Teacher samples - linear_vp/{dataset}/seed{seed}/T={T_last:.2f}",
            outdir / "teacher_samples.png",
        )

    train_start = time.perf_counter()
    model = train_mirafzali_skorokhod_net(
        t_tr,
        x_tr,
        s_tr,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        hidden=hidden,
        n_blocks=n_blocks,
        device=device,
    )
    train_seconds = time.perf_counter() - train_start

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "dataset": dataset,
            "sde_type": "linear_vp",
            "cfg": cfg.__dict__,
            "times": list(times),
            "seed": seed,
        },
        outdir / "model.pt",
    )

    reverse_start = time.perf_counter()
    samples = reverse_sample_linear(
        model=model.eval().to(device),
        n_samples=n_samples_reverse,
        sde_config=cfg,
        n_steps=n_steps_rev,
        device=device,
    )
    reverse_seconds = time.perf_counter() - reverse_start

    nan_mask = torch.isnan(samples).any(dim=1)
    nan_rate = float(nan_mask.float().mean().item())
    clean = samples[~nan_mask].detach().cpu().numpy()
    print(f"  {len(clean):,} clean samples  (nan_rate={nan_rate:.3f})", flush=True)

    centers_np = None
    if dataset == "8gmm":
        _, centers_t = sample_8gmm(1, device="cpu")
        centers_np = centers_t.numpy()

    metrics = compute_metrics_nl(
        clean,
        nan_rate,
        dataset_name=dataset,
        centers_np=centers_np,
        rng=np.random.default_rng(seed),
    )

    # Malliavin diagnostics at the largest time point.
    H_last = sim_cache[-1][2]
    H_norm = torch.linalg.norm(H_last, dim=1)
    metrics.update(
        {
            "var_H": float(H_last.var(unbiased=False).item()),
            "mean_H_norm": float(H_norm.mean().item()),
            "var_H_norm": float(H_norm.var(unbiased=False).item()),
            "sde_type": "linear_vp",
            "sim_seconds": sim_seconds,
            "train_seconds": train_seconds,
            "reverse_seconds": reverse_seconds,
            "total_seconds": time.perf_counter() - run_start,
        }
    )

    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    sampler = get_sampler(dataset)
    result_r = sampler(5_000, device="cpu")
    ref_np = (result_r[0] if isinstance(result_r, tuple) else result_r).numpy()
    _plot_scatter(
        clean[:5_000],
        f"Reverse samples - linear_vp/{dataset}/seed{seed}",
        outdir / "reverse_samples.png",
        ref_np=ref_np,
    )

    print(
        f"  mmd={metrics['mmd_rbf']:.5f}  sw={metrics['sliced_wasserstein']:.5f}",
        flush=True,
    )

    return metrics


def run_linear_vp_multiseed(
    dataset: str = "swissroll",
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    cfg: VPConfig = VPConfig(beta_min=1.0, beta_max=25.0, T=1.0),
    times: Sequence[float] = TIMES,
    hidden: int = 2048,
    n_blocks: int = 6,
    n_paths: int = 25_000,
    n_epochs: int = 5_000,
    batch_size: int = 4_096,
    lr: float = 2e-4,
    n_samples_reverse: int = 20_000,
    n_steps_rev: int = 1_000,
    outbase: str = "results/linear_vs_nonlinear_swissroll_lowt_stationary_5seed",
    device: Optional[str] = None,
) -> dict:
    outdir = Path(outbase)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in seeds:
        metrics = run_linear_vp_seed(
            dataset=dataset,
            seed=seed,
            cfg=cfg,
            times=times,
            hidden=hidden,
            n_blocks=n_blocks,
            n_paths=n_paths,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            n_samples_reverse=n_samples_reverse,
            n_steps_rev=n_steps_rev,
            outbase=outbase,
            device=device,
        )
        rows.append(
            {
                "seed": seed,
                "config_key": "linear_vp",
                "sde_type": "linear_vp",
                "mmd_rbf": metrics.get("mmd_rbf"),
                "sliced_wasserstein": metrics.get("sliced_wasserstein"),
                "nan_rate": metrics.get("nan_rate"),
                "var_H": metrics.get("var_H"),
                "mean_H_norm": metrics.get("mean_H_norm"),
                "var_H_norm": metrics.get("var_H_norm"),
                "sim_seconds": metrics.get("sim_seconds"),
                "train_seconds": metrics.get("train_seconds"),
                "reverse_seconds": metrics.get("reverse_seconds"),
                "total_seconds": metrics.get("total_seconds"),
            }
        )

    raw_json_path = outdir / "raw_results.json"
    with open(raw_json_path, "w") as f:
        json.dump(rows, f, indent=2)

    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(outdir / "raw_results.csv", index=False)

    summary = {
        "config_key": "linear_vp",
        "sde_type": "linear_vp",
        "dataset": dataset,
        "n_seeds": int(len(raw_df)),
        "mmd_mean": float(raw_df["mmd_rbf"].mean()),
        "mmd_std": float(raw_df["mmd_rbf"].std(ddof=1)) if len(raw_df) > 1 else None,
        "sw_mean": float(raw_df["sliced_wasserstein"].mean()),
        "sw_std": float(raw_df["sliced_wasserstein"].std(ddof=1)) if len(raw_df) > 1 else None,
        "nan_rate_mean": float(raw_df["nan_rate"].mean()),
        "var_H_mean": float(raw_df["var_H"].mean()),
        "mean_H_norm_mean": float(raw_df["mean_H_norm"].mean()),
        "var_H_norm_mean": float(raw_df["var_H_norm"].mean()),
        "sim_seconds_mean": float(raw_df["sim_seconds"].mean()),
        "train_seconds_mean": float(raw_df["train_seconds"].mean()),
        "reverse_seconds_mean": float(raw_df["reverse_seconds"].mean()),
        "total_seconds_mean": float(raw_df["total_seconds"].mean()),
    }

    summary_rows = [summary]
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)
    pd.DataFrame(summary_rows).to_csv(outdir / "summary.csv", index=False)

    print(f"\nSaved: {raw_json_path}", flush=True)
    print(f"Saved: {outdir / 'summary.json'}", flush=True)

    return {
        "raw": rows,
        "summary": summary_rows,
    }


if __name__ == "__main__":
    run_linear_vp_multiseed(
        dataset="swissroll",
        seeds=[0],
        cfg=VPConfig(beta_min=1.0, beta_max=25.0, T=1.0),
        times=TIMES,
        hidden=2048,
        n_blocks=6,
        n_paths=25_000,
        n_epochs=5_000,
        batch_size=4_096,
        lr=2e-4,
        n_samples_reverse=20_000,
        n_steps_rev=1_000,
        outbase="results/linear_vs_nonlinear_swissroll_lowt_stationary_5seed",
    )
