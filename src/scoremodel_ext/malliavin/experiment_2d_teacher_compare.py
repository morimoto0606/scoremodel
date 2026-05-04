"""
Spec 02: 2D NW Teacher Generation

Compare four teacher methods on the 8-GMM reverse-sampling task:
  - binned  : 2D histogram bin averages
  - nw      : Nadaraya-Watson Gaussian kernel regression
  - knn_nw  : kNN-adaptive NW kernel regression
  - raw     : direct per-path (X_T, H) pairs (subsampled)

For each method:
  1. Build training dataset from shared forward simulations.
  2. Train a TimeScoreMLP2D score model.
  3. Run reverse sampling.
  4. Compute mode_coverage, nearest_mode_dist, MMD, Wasserstein, nan_rate.
  5. Save metrics.json and diagnostic plots.

Output structure:
  results/2d_teacher_compare/{binned,nw,knn_nw,raw}/
  results/2d_teacher_compare/metrics_summary.json
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")

try:
    from .models import TimeScoreMLP2D
    from .sde_2d import (
        sample_8gmm,
        simulate_2d_malliavin_ito,
        bin_teacher_2d,
        nw_teacher_2d,
        knn_nw_teacher_2d,
    )
    from .experiment_2d_time_reverse_sampling import reverse_sample
except ImportError:
    from models import TimeScoreMLP2D
    from sde_2d import (
        sample_8gmm,
        simulate_2d_malliavin_ito,
        bin_teacher_2d,
        nw_teacher_2d,
        knn_nw_teacher_2d,
    )
    from experiment_2d_time_reverse_sampling import reverse_sample


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

METHODS = ("binned", "nw", "knn_nw", "raw")
TIMES = [0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00]

# Bandwidth sweep grids
NW_BANDWIDTHS = [0.05, 0.08, 0.12, 0.18, 0.25]
KNN_K_VALUES  = [64, 128, 256, 512]
KNN_BW_SCALES = [0.5, 0.75, 1.0]

_DEFAULT_SIM = dict(
    n_paths=250_000,
    n_steps=120,
    sigma_min=0.15,
    sigma_max=1.20,
    gamma_reg=1e-3,
)


# ──────────────────────────────────────────────────────────────────────────────
# Teacher application
# ──────────────────────────────────────────────────────────────────────────────

def apply_teacher(
    method,
    X_T,
    H,
    n_raw=50_000,
    knn_k=500,
    n_bins=80,
    min_count=25,
):
    """
    Given terminal samples X_T (n,2) and per-path weights H (n,2),
    apply the chosen teacher method.

    Returns (pts, scores, counts) with shapes (m,2), (m,2), (m,).
    """
    if method == "binned":
        return bin_teacher_2d(X_T, H, n_bins=n_bins, min_count=min_count)

    elif method in ("nw", "knn_nw"):
        # Use the binned grid as query locations for consistent comparison.
        pts, _, cc = bin_teacher_2d(X_T, H, n_bins=n_bins, min_count=min_count)
        if len(pts) == 0:
            sc = pts.clone()
        elif method == "nw":
            sc = nw_teacher_2d(X_T, H, pts)
        else:
            sc = knn_nw_teacher_2d(X_T, H, pts, k=knn_k)
        return pts, sc, cc

    elif method == "raw":
        n = X_T.shape[0]
        idx = torch.randperm(n, device=X_T.device)[:min(n_raw, n)]
        pts = X_T[idx]
        sc = H[idx]
        cc = torch.ones(pts.shape[0], device=X_T.device)
        return pts, sc, cc

    else:
        raise ValueError(f"Unknown teacher method: {method!r}")


# ──────────────────────────────────────────────────────────────────────────────
# Sweep-specific teacher application
# ──────────────────────────────────────────────────────────────────────────────

def apply_teacher_sweep(
    family,
    X_T,
    H,
    bandwidth=None,
    k=None,
    bandwidth_scale=1.0,
    n_bins=80,
    min_count=25,
):
    """
    Parametric teacher for bandwidth sweeps.  Query points are always the
    binned grid (consistent across configs).

    family  : "nw" or "knn_nw"
    Returns (pts, sc, cc) with shapes (m,2), (m,2), (m,).
    """
    if family not in ("nw", "knn_nw"):
        raise ValueError(f"Unknown sweep family: {family!r}")

    pts, _, cc = bin_teacher_2d(X_T, H, n_bins=n_bins, min_count=min_count)
    if pts.shape[0] == 0:
        return pts, pts.clone(), cc

    if family == "nw":
        sc = nw_teacher_2d(X_T, H, pts, bandwidth=bandwidth)
    else:  # knn_nw
        sc = knn_nw_teacher_2d(X_T, H, pts, k=k, bandwidth_scale=bandwidth_scale)

    return pts, sc, cc


# ──────────────────────────────────────────────────────────────────────────────
# Shared forward simulation cache
# ──────────────────────────────────────────────────────────────────────────────

def simulate_all_times(times, device, **sim_kw):
    """
    Simulate the forward SDE once per T.

    Returns list of (T, X_T, H) tuples (tensors live on *device*).
    """
    full_sim = {**_DEFAULT_SIM, **sim_kw}
    cache = []
    for T in times:
        print(f"  [simulate] T={T:.2f} …", flush=True)
        X_T, H, _, stats = simulate_2d_malliavin_ito(T=T, device=device, **full_sim)
        print(
            f"    gamma_min_eig={stats['gamma_min_eig']:.3e}  "
            f"H_norm_mean={stats['H_norm_mean']:.3f}",
            flush=True,
        )
        cache.append((float(T), X_T, H))
    return cache


def build_sweep_dataset_from_cache(
    sim_cache,
    family,
    bandwidth=None,
    k=None,
    bandwidth_scale=1.0,
    n_bins=80,
    min_count=25,
):
    """
    Apply a parametric teacher to pre-simulated (T, X_T, H) data.

    Returns (t, x, s, c) tensors, or None if no valid points found.
    """
    t_list, x_list, s_list, c_list = [], [], [], []
    device = sim_cache[0][1].device

    for T, X_T, H in sim_cache:
        pts, sc, cc = apply_teacher_sweep(
            family, X_T, H,
            bandwidth=bandwidth,
            k=k,
            bandwidth_scale=bandwidth_scale,
            n_bins=n_bins,
            min_count=min_count,
        )
        if pts.shape[0] == 0:
            continue
        t_list.append(torch.full((pts.shape[0],), T, device=device))
        x_list.append(pts)
        s_list.append(sc)
        c_list.append(cc)

    if not t_list:
        return None
    return (
        torch.cat(t_list),
        torch.cat(x_list),
        torch.cat(s_list),
        torch.cat(c_list),
    )


def _build_baseline_dataset_from_cache(
    sim_cache,
    method,
    n_bins=80,
    min_count=25,
    n_raw=50_000,
):
    """
    Apply a baseline teacher ("raw" or "binned") to pre-simulated
    (T, X_T, H) data.

    Returns (t, x, s, c) tensors, or None if no valid points found.
    """
    if method not in ("raw", "binned"):
        raise ValueError(
            f"_build_baseline_dataset_from_cache only supports 'raw' or 'binned', "
            f"got {method!r}"
        )
    t_list, x_list, s_list, c_list = [], [], [], []
    device = sim_cache[0][1].device

    for T, X_T, H in sim_cache:
        pts, sc, cc = apply_teacher(
            method, X_T, H,
            n_raw=n_raw,
            n_bins=n_bins,
            min_count=min_count,
        )
        if pts.shape[0] == 0:
            continue
        t_list.append(torch.full((pts.shape[0],), T, device=device))
        x_list.append(pts)
        s_list.append(sc)
        c_list.append(cc)

    if not t_list:
        return None
    return (
        torch.cat(t_list),
        torch.cat(x_list),
        torch.cat(s_list),
        torch.cat(c_list),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Teacher-field plot (raw teacher data, no MLP involved)
# ──────────────────────────────────────────────────────────────────────────────

def _plot_teacher_field(pts_np, sc_np, T, title, path, lim=3.2):
    """Quiver of raw per-bin teacher score estimates at time T."""
    plt.figure(figsize=(7, 7))
    plt.quiver(
        pts_np[:, 0], pts_np[:, 1],
        sc_np[:, 0],  sc_np[:, 1],
        angles="xy", scale_units="xy", scale=22, width=0.002, alpha=0.75,
    )
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.gca().set_aspect("equal")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    print(f"saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Single sweep-config runner
# ──────────────────────────────────────────────────────────────────────────────

def run_sweep_config(
    label,
    family,
    sim_cache,
    times,
    outdir,
    bandwidth=None,
    k=None,
    bandwidth_scale=1.0,
    n_bins=80,
    min_count=25,
    device="cuda",
    n_epochs=8_000,
    batch_size=4_096,
    lr=2e-4,
    n_samples_reverse=30_000,
    n_steps_reverse=800,
    sigma_min=0.15,
    sigma_max=1.20,
):
    """
    Run the full train → reverse → metrics pipeline for one sweep config.

    Returns the metrics dict or None if no training data.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Sweep config: {label}")
    print(f"{'='*60}", flush=True)

    # ── build dataset ──────────────────────────────────────────────────────
    dataset = build_sweep_dataset_from_cache(
        sim_cache, family,
        bandwidth=bandwidth, k=k, bandwidth_scale=bandwidth_scale,
        n_bins=n_bins, min_count=min_count,
    )
    if dataset is None:
        print(f"  [{label}] no valid training points, skipping")
        return None

    t, x, s, c = dataset
    print(f"  training points: {t.shape[0]:,}", flush=True)

    # ── teacher-field plots (raw teacher, per T) ───────────────────────────
    for T, X_T, H in sim_cache:
        pts, sc_t, _ = apply_teacher_sweep(
            family, X_T, H,
            bandwidth=bandwidth, k=k, bandwidth_scale=bandwidth_scale,
            n_bins=n_bins, min_count=min_count,
        )
        if pts.shape[0] == 0:
            continue
        _plot_teacher_field(
            pts.cpu().numpy(), sc_t.cpu().numpy(),
            T,
            f"Teacher field — {label} — T={T:.2f}",
            outdir / f"teacher_field_T_{T:.2f}.png",
        )

    # ── train ──────────────────────────────────────────────────────────────
    model = train_time_mlp(
        t, x, s, c,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
    )
    torch.save(
        {"model_state_dict": model.state_dict(), "label": label, "times": times},
        outdir / "model.pt",
    )

    # ── learned score-field plots ──────────────────────────────────────────
    for T in times:
        _plot_score_field(
            model, T,
            outdir / f"score_field_T_{T:.2f}.png",
            device=device,
        )

    # ── reverse sampling ───────────────────────────────────────────────────
    T_max = max(times)
    final_x, _, _, centers, _ = reverse_sample(
        model,
        n_samples=n_samples_reverse,
        T=T_max,
        n_steps=n_steps_reverse,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        device=device,
    )

    nan_mask = torch.isnan(final_x).any(dim=1)
    nan_rate = float(nan_mask.float().mean().item())
    final_x_clean = final_x[~nan_mask]

    samples_np = final_x_clean.numpy()
    centers_np = centers.numpy()

    _plot_scatter(
        samples_np, centers_np,
        f"Reverse samples — {label}",
        outdir / "reverse_samples.png",
    )

    # ── metrics ────────────────────────────────────────────────────────────
    metrics = compute_metrics(samples_np, centers_np, nan_rate=nan_rate)
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(
        f"  [{label}]  coverage={metrics['mode_coverage']['coverage_fraction']:.2f}  "
        f"mmd={metrics['mmd_rbf']:.4f}  sw={metrics['sliced_wasserstein']:.4f}",
        flush=True,
    )

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Bandwidth sweep runner
# ──────────────────────────────────────────────────────────────────────────────

def run_bandwidth_sweep(
    times=TIMES,
    nw_bandwidths=NW_BANDWIDTHS,
    knn_k_values=KNN_K_VALUES,
    knn_bw_scales=KNN_BW_SCALES,
    device=None,
    n_paths=250_000,
    n_steps=120,
    sigma_min=0.15,
    sigma_max=1.20,
    gamma_reg=1e-3,
    n_bins=80,
    min_count=25,
    n_epochs=8_000,
    batch_size=4_096,
    lr=2e-4,
    n_samples_reverse=30_000,
    n_steps_reverse=800,
    outbase="results/2d_teacher_compare/sweep",
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    base = Path(outbase)
    base.mkdir(parents=True, exist_ok=True)

    # ── shared forward simulation (once per T) ─────────────────────────────
    print("\n[1/3] Simulating forward SDE …")
    sim_cache = simulate_all_times(
        times, device,
        n_paths=n_paths,
        n_steps=n_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        gamma_reg=gamma_reg,
    )

    # ── build sweep configs ────────────────────────────────────────────────
    sweep_configs = []
    for h in nw_bandwidths:
        label = f"nw_h{h:.2f}"
        sweep_configs.append(dict(
            label=label,
            family="nw",
            bandwidth=h,
            k=None,
            bandwidth_scale=1.0,
        ))
    for k in knn_k_values:
        for bw in knn_bw_scales:
            label = f"knn_k{k}_s{bw:.2f}"
            sweep_configs.append(dict(
                label=label,
                family="knn_nw",
                bandwidth=None,
                k=k,
                bandwidth_scale=bw,
            ))

    # ── run each config ────────────────────────────────────────────────────
    print(f"\n[2/3] Running {len(sweep_configs)} sweep configs …")
    all_results = {}  # label -> {config, metrics}

    for cfg in sweep_configs:
        label = cfg["label"]
        m = run_sweep_config(
            label=label,
            family=cfg["family"],
            sim_cache=sim_cache,
            times=times,
            outdir=base / label,
            bandwidth=cfg["bandwidth"],
            k=cfg["k"],
            bandwidth_scale=cfg["bandwidth_scale"],
            n_bins=n_bins,
            min_count=min_count,
            device=device,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            n_samples_reverse=n_samples_reverse,
            n_steps_reverse=n_steps_reverse,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
        if m is not None:
            all_results[label] = {
                "config": {k2: v for k2, v in cfg.items() if k2 != "label"},
                "metrics": m,
            }

    # ── best by MMD per family ─────────────────────────────────────────────
    print("\n[3/3] Computing best configs by MMD …")

    best_by_mmd: dict = {}
    for family in ("nw", "knn_nw"):
        candidates = {
            label: v
            for label, v in all_results.items()
            if v["config"]["family"] == family
        }
        if not candidates:
            continue
        best_label = min(candidates, key=lambda l: candidates[l]["metrics"]["mmd_rbf"])
        best_entry = candidates[best_label]
        best_by_mmd[family] = {
            "label": best_label,
            "config": best_entry["config"],
            **{k2: v for k2, v in best_entry["metrics"].items()
               if k2 not in ("mode_coverage", "nearest_mode_dist")},
            "mode_coverage_fraction": best_entry["metrics"]["mode_coverage"]["coverage_fraction"],
            "nearest_mode_dist_mean": best_entry["metrics"]["nearest_mode_dist"]["mean"],
        }
        print(
            f"  best {family}: {best_label}  "
            f"mmd={best_entry['metrics']['mmd_rbf']:.4f}"
        )

    best_path = base / "best_by_mmd.json"
    with open(best_path, "w") as f:
        json.dump(best_by_mmd, f, indent=2)
    print(f"Best configs saved → {best_path}")

    # ── full sweep summary JSON ────────────────────────────────────────────
    summary_path = base / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Full sweep summary → {summary_path}")

    # ── sweep metric comparison plots ─────────────────────────────────────
    _plot_sweep_summary(all_results, base)

    return best_by_mmd


def _plot_sweep_summary(all_results, outdir):
    """Line plots of MMD and coverage vs bandwidth for each family."""
    for family in ("nw", "knn_nw"):
        entries = {
            label: v
            for label, v in all_results.items()
            if v["config"]["family"] == family
        }
        if not entries:
            continue

        if family == "nw":
            xs = [v["config"]["bandwidth"] for v in entries.values()]
            x_label = "bandwidth h"
            xs_sorted = sorted(set(xs))
            mmd_vals = [
                next(v["metrics"]["mmd_rbf"] for l, v in entries.items()
                     if v["config"]["bandwidth"] == h)
                for h in xs_sorted
            ]
            cov_vals = [
                next(v["metrics"]["mode_coverage"]["coverage_fraction"]
                     for l, v in entries.items()
                     if v["config"]["bandwidth"] == h)
                for h in xs_sorted
            ]
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            ax1.plot(xs_sorted, mmd_vals, "o-")
            ax1.set_xlabel(x_label)
            ax1.set_title("MMD vs h (NW)")
            ax2.plot(xs_sorted, cov_vals, "o-")
            ax2.set_xlabel(x_label)
            ax2.set_title("Mode coverage vs h (NW)")
            plt.tight_layout()
            plt.savefig(outdir / "sweep_nw.png", dpi=180)
            plt.close()
            print(f"saved: {outdir / 'sweep_nw.png'}")

        else:  # knn_nw — one line per bw_scale
            bw_scales = sorted(set(
                v["config"]["bandwidth_scale"] for v in entries.values()
            ))
            k_vals = sorted(set(v["config"]["k"] for v in entries.values()))

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            for bw in bw_scales:
                sub = {
                    l: v for l, v in entries.items()
                    if v["config"]["bandwidth_scale"] == bw
                }
                ks = sorted(sub, key=lambda l: sub[l]["config"]["k"])
                mmd_line = [sub[l]["metrics"]["mmd_rbf"] for l in ks]
                cov_line = [
                    sub[l]["metrics"]["mode_coverage"]["coverage_fraction"]
                    for l in ks
                ]
                k_line = [sub[l]["config"]["k"] for l in ks]
                ax1.plot(k_line, mmd_line, "o-", label=f"scale={bw:.2f}")
                ax2.plot(k_line, cov_line, "o-", label=f"scale={bw:.2f}")
            ax1.set_xlabel("k")
            ax1.set_title("MMD vs k (kNN-NW)")
            ax1.legend()
            ax2.set_xlabel("k")
            ax2.set_title("Mode coverage vs k (kNN-NW)")
            ax2.legend()
            plt.tight_layout()
            plt.savefig(outdir / "sweep_knn_nw.png", dpi=180)
            plt.close()
            print(f"saved: {outdir / 'sweep_knn_nw.png'}")


# ──────────────────────────────────────────────────────────────────────────────
# Dataset construction (simulate once per T, share across methods)
# ──────────────────────────────────────────────────────────────────────────────

def build_all_datasets(
    times=TIMES,
    device="cuda",
    n_raw=50_000,
    knn_k=500,
    n_bins=80,
    min_count=25,
    **sim_kw,
):
    """
    Simulate forward SDE once per T; apply all four methods to the same paths.

    Returns dict: method -> (t_tensor, x_tensor, s_tensor, c_tensor)
    """
    merged = {m: {"t": [], "x": [], "s": [], "c": []} for m in METHODS}

    full_sim = {**_DEFAULT_SIM, **sim_kw}

    for T in times:
        print(f"  [simulate] T={T:.2f} …", flush=True)
        X_T, H, _, stats = simulate_2d_malliavin_ito(
            T=T,
            device=device,
            **full_sim,
        )
        print(
            f"    gamma_min_eig={stats['gamma_min_eig']:.3e}  "
            f"H_norm_mean={stats['H_norm_mean']:.3f}",
            flush=True,
        )

        for method in METHODS:
            pts, sc, cc = apply_teacher(
                method, X_T, H,
                n_raw=n_raw,
                knn_k=knn_k,
                n_bins=n_bins,
                min_count=min_count,
            )
            if pts.shape[0] == 0:
                print(f"    [{method}] T={T:.2f}: no valid points, skipping", flush=True)
                continue
            t_vec = torch.full((pts.shape[0],), float(T), device=device)
            merged[method]["t"].append(t_vec)
            merged[method]["x"].append(pts)
            merged[method]["s"].append(sc)
            merged[method]["c"].append(cc)

    result = {}
    for m, d in merged.items():
        if not d["t"]:
            continue
        result[m] = (
            torch.cat(d["t"]),
            torch.cat(d["x"]),
            torch.cat(d["s"]),
            torch.cat(d["c"]),
        )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_time_mlp(
    t,
    x,
    s,
    c,
    n_epochs=8_000,
    batch_size=4_096,
    lr=2e-4,
    device="cuda",
):
    """
    Train TimeScoreMLP2D on (t, x) -> s with per-sample weights c.
    Returns the best model (by weighted MSE on full dataset).
    """
    model = TimeScoreMLP2D().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    w = c / c.mean()

    best = float("inf")
    best_state = None
    n = x.shape[0]

    for ep in range(1, n_epochs + 1):
        idx = torch.randint(0, n, (batch_size,), device=device)
        pred = model(t[idx], x[idx])
        loss = (w[idx, None] * (pred - s[idx]) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if ep % 500 == 0:
            with torch.no_grad():
                full_loss = (w[:, None] * (model(t, x) - s) ** 2).mean()
            if full_loss.item() < best:
                best = full_loss.item()
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                print(f"  *** best updated: {best:.6e}")
            print(f"  epoch={ep:5d}  loss={full_loss.item():.6e}  best={best:.6e}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def _mode_coverage(samples_np, centers_np, threshold=0.01):
    """
    Fraction of modes whose assigned-sample proportion exceeds *threshold*.
    """
    d2 = ((samples_np[:, None, :] - centers_np[None, :, :]) ** 2).sum(-1)
    assign = d2.argmin(axis=1)
    n_modes = centers_np.shape[0]
    counts = np.bincount(assign, minlength=n_modes).astype(float)
    props = counts / counts.sum()
    covered = (props > threshold).sum()
    return {
        "n_covered": int(covered),
        "n_modes": int(n_modes),
        "coverage_fraction": float(covered) / n_modes,
        "min_proportion": float(props.min()),
        "max_proportion": float(props.max()),
        "proportions": props.tolist(),
    }


def _nearest_mode_dist(samples_np, centers_np):
    d2 = ((samples_np[:, None, :] - centers_np[None, :, :]) ** 2).sum(-1)
    nearest = np.sqrt(d2.min(axis=1))
    return {
        "mean": float(nearest.mean()),
        "std": float(nearest.std()),
        "median": float(np.median(nearest)),
    }


def _mmd_rbf(X, Y, n_sub=2_000, sigma=1.0, rng=None):
    """Unbiased RBF-MMD^2 estimate (subsampled)."""
    if rng is None:
        rng = np.random.default_rng(42)
    ix = rng.choice(len(X), min(n_sub, len(X)), replace=False)
    iy = rng.choice(len(Y), min(n_sub, len(Y)), replace=False)
    Xs, Ys = X[ix], Y[iy]

    def gram(A, B):
        d2 = ((A[:, None] - B[None]) ** 2).sum(-1)
        return np.exp(-d2 / (2 * sigma ** 2))

    Kxx = gram(Xs, Xs)
    Kyy = gram(Ys, Ys)
    Kxy = gram(Xs, Ys)
    n, m = len(Xs), len(Ys)
    val = (
        (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
        + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
        - 2 * Kxy.mean()
    )
    return float(val)


def _sliced_wasserstein(X, Y, n_projections=200, rng=None):
    """Approximate 2D Wasserstein-1 via random projections."""
    if rng is None:
        rng = np.random.default_rng(42)
    d = X.shape[1]
    total = 0.0
    for _ in range(n_projections):
        v = rng.standard_normal(d)
        v /= np.linalg.norm(v)
        px = np.sort(X @ v)
        py = np.sort(Y @ v)
        n, m = len(px), len(py)
        if n != m:
            idx = np.linspace(0, n - 1, m)
            px = np.interp(idx, np.arange(n), px)
        total += float(np.mean(np.abs(px - py)))
    return total / n_projections


def compute_metrics(samples_np, centers_np, nan_rate=0.0, n_ref=30_000, rng=None):
    """
    Compute all Spec 02 metrics.

    samples_np  : (n, 2) numpy array of reverse-sampled points
    centers_np  : (8, 2) numpy array of true mode centres
    nan_rate    : fraction of samples discarded due to NaN / overflow
    n_ref       : number of fresh reference samples drawn from 8-GMM
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Reference samples from the 8-GMM (radius=2, std=0.08)
    radius = 2.0
    std_gmm = 0.08
    angles_ref = np.linspace(0, 2 * math.pi, 9)[:-1]  # 8 equally-spaced angles
    c_ref = radius * np.stack([np.cos(angles_ref), np.sin(angles_ref)], axis=1)
    idx_ref = rng.integers(0, 8, size=n_ref)
    ref_np = c_ref[idx_ref] + std_gmm * rng.standard_normal((n_ref, 2))

    mc = _mode_coverage(samples_np, centers_np)
    nd = _nearest_mode_dist(samples_np, centers_np)
    mmd_val = _mmd_rbf(samples_np, ref_np, rng=rng)
    sw_val = _sliced_wasserstein(samples_np, ref_np, rng=rng)

    return {
        "mode_coverage": mc,
        "nearest_mode_dist": nd,
        "mmd_rbf": mmd_val,
        "sliced_wasserstein": sw_val,
        "nan_rate": float(nan_rate),
        "n_samples": int(len(samples_np)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────

def _plot_scatter(x_np, centers_np, title, path, lim=3.2):
    plt.figure(figsize=(7, 7))
    plt.scatter(x_np[:, 0], x_np[:, 1], s=2, alpha=0.22, label="samples")
    plt.scatter(
        centers_np[:, 0], centers_np[:, 1],
        s=90, marker="x", c="red", label="modes",
    )
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.gca().set_aspect("equal")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    print(f"saved: {path}")


def _plot_score_field(model, T, path, device, lim=3.2, n=40):
    xs = torch.linspace(-lim, lim, n, device=device)
    Xg, Yg = torch.meshgrid(xs, xs, indexing="xy")
    grid = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1)
    tt = torch.full((grid.shape[0],), T, device=device)
    with torch.no_grad():
        pred = model(tt, grid)
    plt.figure(figsize=(7, 7))
    plt.quiver(
        grid[:, 0].cpu(), grid[:, 1].cpu(),
        pred[:, 0].cpu(), pred[:, 1].cpu(),
        angles="xy", scale_units="xy", scale=22, width=0.002,
    )
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.gca().set_aspect("equal")
    plt.title(f"Score field T={T:.2f}")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    print(f"saved: {path}")


def plot_metrics_summary(all_metrics, outdir):
    """Bar chart comparing scalar metrics across methods."""
    methods = list(all_metrics.keys())
    keys = [
        ("mode_coverage", "coverage_fraction", "Mode coverage (fraction)"),
        ("nearest_mode_dist", "mean", "Mean nearest-mode distance"),
        ("mmd_rbf", None, "MMD (RBF)"),
        ("sliced_wasserstein", None, "Sliced Wasserstein"),
    ]

    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 5))
    for ax, (top_key, sub_key, label) in zip(axes, keys):
        vals = []
        for m in methods:
            v = all_metrics[m].get(top_key, 0.0)
            if sub_key is not None and isinstance(v, dict):
                v = v.get(sub_key, 0.0)
            vals.append(float(v))
        ax.bar(methods, vals)
        ax.set_title(label)
        ax.set_xticklabels(methods, rotation=15, ha="right")
    plt.tight_layout()
    path = outdir / "metrics_summary.png"
    plt.savefig(path, dpi=180)
    plt.close()
    print(f"saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Per-method runner
# ──────────────────────────────────────────────────────────────────────────────

def run_one_method(
    method,
    t, x, s, c,
    times,
    outdir,
    device="cuda",
    n_epochs=8_000,
    batch_size=4_096,
    lr=2e-4,
    n_samples_reverse=30_000,
    n_steps_reverse=800,
    sigma_min=0.15,
    sigma_max=1.20,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Method: {method}  ({t.shape[0]:,} training points)")
    print(f"{'='*60}", flush=True)

    # ── train ──────────────────────────────────────────────────────────────
    model = train_time_mlp(
        t, x, s, c,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
    )
    torch.save(
        {"model_state_dict": model.state_dict(), "method": method, "times": times},
        outdir / "model.pt",
    )

    # ── score-field plots ──────────────────────────────────────────────────
    for T in times:
        _plot_score_field(
            model, T,
            outdir / f"score_field_T_{T:.2f}.png",
            device=device,
        )

    # ── reverse sampling ───────────────────────────────────────────────────
    T_max = max(times)
    final_x, _, _, centers, _ = reverse_sample(
        model,
        n_samples=n_samples_reverse,
        T=T_max,
        n_steps=n_steps_reverse,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        device=device,
    )

    # NaN bookkeeping (reverse_sample already clamps, but count any remaining)
    nan_mask = torch.isnan(final_x).any(dim=1)
    nan_rate = float(nan_mask.float().mean().item())
    final_x_clean = final_x[~nan_mask]

    samples_np = final_x_clean.numpy()
    centers_np = centers.numpy()

    _plot_scatter(
        samples_np, centers_np,
        f"Reverse samples — {method}",
        outdir / "reverse_samples.png",
    )

    # ── metrics ────────────────────────────────────────────────────────────
    metrics = compute_metrics(samples_np, centers_np, nan_rate=nan_rate)
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  [{method}] metrics saved → {outdir / 'metrics.json'}")
    print(
        f"  coverage={metrics['mode_coverage']['coverage_fraction']:.2f}  "
        f"mmd={metrics['mmd_rbf']:.4f}  "
        f"sw={metrics['sliced_wasserstein']:.4f}",
        flush=True,
    )

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Main run
# ──────────────────────────────────────────────────────────────────────────────

def run(
    methods=METHODS,
    times=TIMES,
    device=None,
    n_paths=250_000,
    n_steps=120,
    sigma_min=0.15,
    sigma_max=1.20,
    gamma_reg=1e-3,
    n_raw=50_000,
    knn_k=500,
    n_bins=80,
    min_count=25,
    n_epochs=8_000,
    batch_size=4_096,
    lr=2e-4,
    n_samples_reverse=30_000,
    n_steps_reverse=800,
    outbase="results/2d_teacher_compare",
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    base = Path(outbase)
    base.mkdir(parents=True, exist_ok=True)

    # ── shared forward simulation ──────────────────────────────────────────
    print("\n[1/3] Building training datasets …")
    datasets = build_all_datasets(
        times=times,
        device=device,
        n_raw=n_raw,
        knn_k=knn_k,
        n_bins=n_bins,
        min_count=min_count,
        n_paths=n_paths,
        n_steps=n_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        gamma_reg=gamma_reg,
    )

    # ── per-method train + evaluate ────────────────────────────────────────
    print("\n[2/3] Training and evaluating each method …")
    all_metrics = {}
    for method in methods:
        if method not in datasets:
            print(f"  [{method}] no dataset produced, skipping")
            continue
        t, x, s, c = datasets[method]
        m = run_one_method(
            method=method,
            t=t, x=x, s=s, c=c,
            times=times,
            outdir=base / method,
            device=device,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            n_samples_reverse=n_samples_reverse,
            n_steps_reverse=n_steps_reverse,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
        all_metrics[method] = m

    # ── summary ───────────────────────────────────────────────────────────
    print("\n[3/3] Saving summary …")
    summary_path = base / "metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Summary → {summary_path}")

    plot_metrics_summary(all_metrics, base)


# ──────────────────────────────────────────────────────────────────────────────
# Multi-seed evaluation of top-N sweep configs
# ──────────────────────────────────────────────────────────────────────────────

def _aggregate_seeds(per_seed_metrics):
    """
    Given a list of single-seed metric dicts, compute mean and std of
    mmd_rbf and sliced_wasserstein.

    per_seed_metrics : list of dicts with keys 'mmd_rbf', 'sliced_wasserstein'
    Returns dict with keys: mmd_rbf_mean, mmd_rbf_std,
                            sliced_wasserstein_mean, sliced_wasserstein_std
    """
    mmds = np.array([m["mmd_rbf"] for m in per_seed_metrics])
    sws  = np.array([m["sliced_wasserstein"] for m in per_seed_metrics])
    return {
        "mmd_rbf_mean": float(mmds.mean()),
        "mmd_rbf_std":  float(mmds.std(ddof=1) if len(mmds) > 1 else 0.0),
        "sliced_wasserstein_mean": float(sws.mean()),
        "sliced_wasserstein_std":  float(sws.std(ddof=1) if len(sws) > 1 else 0.0),
    }


def _run_single_seed(
    config,
    times,
    seed,
    outdir,
    device,
    n_paths,
    n_steps,
    sigma_min,
    sigma_max,
    gamma_reg,
    n_bins,
    min_count,
    n_epochs,
    batch_size,
    lr,
    n_samples_reverse,
    n_steps_reverse,
):
    """
    Full simulate → teacher → train → reverse → metrics pipeline for one seed.
    Returns metrics dict or None on failure.
    """
    # Seed all randomness sources
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── simulate ──────────────────────────────────────────────────────────
    sim_cache = simulate_all_times(
        times, device,
        n_paths=n_paths,
        n_steps=n_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        gamma_reg=gamma_reg,
    )

    # ── build dataset ──────────────────────────────────────────────────────
    _method = config.get("method")
    if _method in ("raw", "binned"):
        dataset = _build_baseline_dataset_from_cache(
            sim_cache, _method,
            n_bins=n_bins, min_count=min_count,
        )
    else:
        dataset = build_sweep_dataset_from_cache(
            sim_cache,
            config["family"],
            bandwidth=config.get("bandwidth"),
            k=config.get("k"),
            bandwidth_scale=config.get("bandwidth_scale", 1.0),
            n_bins=n_bins,
            min_count=min_count,
        )
    if dataset is None:
        print(f"  [seed={seed}] no valid training data, skipping", flush=True)
        return None

    t, x, s, c = dataset

    # ── train ──────────────────────────────────────────────────────────────
    model = train_time_mlp(
        t, x, s, c,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
    )

    # ── reverse sampling ───────────────────────────────────────────────────
    T_max = max(times)
    final_x, _, _, centers, _ = reverse_sample(
        model,
        n_samples=n_samples_reverse,
        T=T_max,
        n_steps=n_steps_reverse,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        device=device,
    )

    nan_mask = torch.isnan(final_x).any(dim=1)
    nan_rate  = float(nan_mask.float().mean().item())
    final_x_clean = final_x[~nan_mask]

    samples_np  = final_x_clean.numpy()
    centers_np  = centers.numpy()

    _plot_scatter(
        samples_np, centers_np,
        f"Reverse samples (seed={seed})",
        outdir / f"reverse_samples_seed{seed}.png",
    )

    # ── metrics ────────────────────────────────────────────────────────────
    metrics = compute_metrics(
        samples_np, centers_np,
        nan_rate=nan_rate,
        rng=np.random.default_rng(seed),   # deterministic metric RNG
    )
    with open(outdir / f"metrics_seed{seed}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(
        f"  [seed={seed}]  coverage={metrics['mode_coverage']['coverage_fraction']:.2f}  "
        f"mmd={metrics['mmd_rbf']:.4f}  sw={metrics['sliced_wasserstein']:.4f}",
        flush=True,
    )
    return metrics


# Baseline configs that are always included in the multi-seed comparison.
_BASELINE_CONFIGS = {
    "baseline_raw":    {"method": "raw"},
    "baseline_binned": {"method": "binned"},
}


def run_top_configs_multiseed(
    sweep_summary_path="results/2d_teacher_compare/sweep/sweep_summary.json",
    n_top=5,
    seeds=(0, 1, 2),
    times=TIMES,
    device=None,
    n_paths=250_000,
    n_steps=120,
    sigma_min=0.15,
    sigma_max=1.20,
    gamma_reg=1e-3,
    n_bins=80,
    min_count=25,
    n_epochs=8_000,
    batch_size=4_096,
    lr=2e-4,
    n_samples_reverse=30_000,
    n_steps_reverse=800,
    outbase="results/2d_teacher_compare/sweep/multiseed",
):
    """
    Load sweep_summary.json, pick top *n_top* configs by mmd_rbf, and re-run
    each—together with the two fixed baseline configs (raw, binned)—with
    multiple random seeds.

    Output
    ------
    {outbase}/{label}/seed{s}/metrics_seed{s}.json   — per-seed raw metrics
    {outbase}/{label}/seed{s}/reverse_samples_seed{s}.png
    {outbase}/{label}/aggregated.json                 — mean/std across seeds
    {outbase}/top{n_top}_multiseed.json               — sweep top-N only
    {outbase}/comparison_multiseed.json               — baselines + top-N
    {outbase}/comparison_multiseed_summary.png        — combined error-bar plot
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    sweep_summary_path = Path(sweep_summary_path)
    if not sweep_summary_path.exists():
        raise FileNotFoundError(f"sweep_summary.json not found: {sweep_summary_path}")

    with open(sweep_summary_path) as f:
        sweep_data = json.load(f)

    # ── select top-N sweep configs by MMD ─────────────────────────────────
    ranked = sorted(
        sweep_data.items(),
        key=lambda kv: kv[1]["metrics"]["mmd_rbf"],
    )
    top_configs = [(label, entry["config"]) for label, entry in ranked[:n_top]]
    print(f"\nTop {n_top} sweep configs by MMD:")
    for label, _ in top_configs:
        print(f"  {label:25s}  mmd={sweep_data[label]['metrics']['mmd_rbf']:.6f}")

    # ── build the ordered run list: baselines first, then top-N ───────────
    run_list = [
        (label, cfg, "baseline")
        for label, cfg in _BASELINE_CONFIGS.items()
    ] + [
        (label, cfg, f"sweep_top{n_top}")
        for label, cfg in top_configs
    ]

    base = Path(outbase)
    base.mkdir(parents=True, exist_ok=True)

    # ── run each config × each seed ────────────────────────────────────────
    full_results = {}   # label -> result entry (all configs)
    baseline_labels = set(_BASELINE_CONFIGS.keys())

    for label, cfg, group in run_list:
        print(f"\n{'='*60}")
        print(f"  Config: {label}  group={group}  ({len(seeds)} seeds)")
        print(f"{'='*60}", flush=True)

        per_seed_metrics = {}
        cfg_outdir = base / label

        for seed in seeds:
            print(f"\n  --- seed={seed} ---", flush=True)
            m = _run_single_seed(
                config=cfg,
                times=times,
                seed=seed,
                outdir=cfg_outdir / f"seed{seed}",
                device=device,
                n_paths=n_paths,
                n_steps=n_steps,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                gamma_reg=gamma_reg,
                n_bins=n_bins,
                min_count=min_count,
                n_epochs=n_epochs,
                batch_size=batch_size,
                lr=lr,
                n_samples_reverse=n_samples_reverse,
                n_steps_reverse=n_steps_reverse,
            )
            if m is not None:
                per_seed_metrics[seed] = m

        if not per_seed_metrics:
            print(f"  [{label}] all seeds failed, skipping aggregation")
            continue

        # ── aggregate ─────────────────────────────────────────────────────
        agg = _aggregate_seeds(list(per_seed_metrics.values()))
        print(
            f"\n  [{label}] AGGREGATE  "
            f"mmd={agg['mmd_rbf_mean']:.4f}±{agg['mmd_rbf_std']:.4f}  "
            f"sw={agg['sliced_wasserstein_mean']:.4f}±{agg['sliced_wasserstein_std']:.4f}",
            flush=True,
        )

        aggregated = {
            "config": cfg,
            "group": group,
            "seeds": list(per_seed_metrics.keys()),
            **agg,
        }
        with open(cfg_outdir / "aggregated.json", "w") as f:
            json.dump(aggregated, f, indent=2)

        full_results[label] = {
            "config": cfg,
            "group": group,
            "per_seed": {
                str(s): {
                    "mmd_rbf": per_seed_metrics[s]["mmd_rbf"],
                    "sliced_wasserstein": per_seed_metrics[s]["sliced_wasserstein"],
                    "mode_coverage_fraction": (
                        per_seed_metrics[s]["mode_coverage"]["coverage_fraction"]
                    ),
                    "nan_rate": per_seed_metrics[s]["nan_rate"],
                }
                for s in per_seed_metrics
            },
            **agg,
        }

    # ── separate views ─────────────────────────────────────────────────────
    sweep_results     = {l: v for l, v in full_results.items()
                         if l not in baseline_labels}
    comparison_results = full_results  # baselines + sweep top-N

    # sweep top-N only (backward-compat)
    top_path = base / f"top{n_top}_multiseed.json"
    with open(top_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nSweep top-{n_top} multi-seed summary → {top_path}")

    # combined comparison
    comparison_path = base / "comparison_multiseed.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison_results, f, indent=2)
    print(f"Comparison (baselines + top-{n_top}) → {comparison_path}")

    # ── plots ──────────────────────────────────────────────────────────────
    _plot_multiseed_summary(
        comparison_results,
        base,
        baseline_labels=baseline_labels,
        title_suffix=f"baselines + top-{n_top}",
        fname="comparison_multiseed_summary.png",
    )

    return full_results


def _plot_multiseed_summary(
    results,
    outdir,
    baseline_labels=None,
    title_suffix="",
    fname="multiseed_summary.png",
):
    """
    Error-bar plots for mmd_rbf and sliced_wasserstein.
    Baseline configs are drawn in grey; sweep configs in the default colour.
    """
    if not results:
        return

    if baseline_labels is None:
        baseline_labels = set()

    labels  = list(results.keys())
    mmd_mu  = [results[l]["mmd_rbf_mean"] for l in labels]
    mmd_sig = [results[l]["mmd_rbf_std"]  for l in labels]
    sw_mu   = [results[l]["sliced_wasserstein_mean"] for l in labels]
    sw_sig  = [results[l]["sliced_wasserstein_std"]  for l in labels]
    colors  = ["#aaaaaa" if l in baseline_labels else "#4c72b0" for l in labels]

    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(8, 2 * len(labels)), 5))

    for ax, mu, sig, ylabel, metric_name in [
        (ax1, mmd_mu, mmd_sig, "MMD²",  "MMD (RBF)"),
        (ax2, sw_mu,  sw_sig,  "SW",    "Sliced Wasserstein"),
    ]:
        bars = ax.bar(x, mu, yerr=sig, capsize=5, alpha=0.85, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        title = f"{metric_name}, mean±std"
        if title_suffix:
            title += f" ({title_suffix})"
        ax.set_title(title)
        ax.set_ylabel(ylabel)

    # legend: grey = baseline, blue = sweep
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#aaaaaa", alpha=0.85, label="baseline"),
        Patch(facecolor="#4c72b0", alpha=0.85, label="sweep"),
    ]
    ax2.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    path = outdir / fname
    plt.savefig(path, dpi=180)
    plt.close()
    print(f"saved: {path}")


if __name__ == "__main__":
    run()
