"""
pytest smoke tests for experiment_2d_teacher_compare (Spec 02).

Run with:
    cd /export/home/ymorimoto/github/scoremodel
    pytest tests/test_teacher_compare_2d.py -v
"""

import math
import json
import numpy as np
import pytest
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scoremodel_ext.malliavin.sde_2d import (
    sample_8gmm,
    simulate_2d_malliavin_ito,
    bin_teacher_2d,
    nw_teacher_2d,
    knn_nw_teacher_2d,
)
from scoremodel_ext.malliavin.experiment_2d_teacher_compare import (
    apply_teacher,
    apply_teacher_sweep,
    simulate_all_times,
    build_sweep_dataset_from_cache,
    _build_baseline_dataset_from_cache,
    build_all_datasets,
    train_time_mlp,
    compute_metrics,
    _aggregate_seeds,
    _run_single_seed,
    _BASELINE_CONFIGS,
    _mode_coverage,
    _nearest_mode_dist,
    _mmd_rbf,
    _sliced_wasserstein,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── tiny forward simulation fixture ──────────────────────────────────────────

@pytest.fixture(scope="module")
def small_sim():
    """Small forward simulation used across multiple tests."""
    X_T, H, centers, stats = simulate_2d_malliavin_ito(
        n_paths=1_000,
        T=0.35,
        n_steps=20,
        sigma=0.45,
        gamma_reg=1e-3,
        device=DEVICE,
    )
    return X_T, H, centers, stats


# ──────────────────────────────────────────────────────────────────────────────
# sde_2d.nw_teacher_2d
# ──────────────────────────────────────────────────────────────────────────────

class TestNWTeacher2D:
    def test_output_shape(self, small_sim):
        X_T, H, _, _ = small_sim
        query = X_T[:50]
        out = nw_teacher_2d(X_T, H, query)
        assert out.shape == (50, 2), f"Expected (50,2), got {out.shape}"

    def test_no_nan(self, small_sim):
        X_T, H, _, _ = small_sim
        query = X_T[:50]
        out = nw_teacher_2d(X_T, H, query)
        assert not torch.isnan(out).any(), "NaN in NW output"

    def test_custom_bandwidth(self, small_sim):
        X_T, H, _, _ = small_sim
        query = X_T[:20]
        out = nw_teacher_2d(X_T, H, query, bandwidth=0.5)
        assert out.shape == (20, 2)
        assert not torch.isnan(out).any()

    def test_batch_size_consistency(self, small_sim):
        """Different batch_sizes should give the same result."""
        X_T, H, _, _ = small_sim
        query = X_T[:30]
        out1 = nw_teacher_2d(X_T, H, query, batch_size=10)
        out2 = nw_teacher_2d(X_T, H, query, batch_size=30)
        assert torch.allclose(out1, out2, atol=1e-5)


# ──────────────────────────────────────────────────────────────────────────────
# sde_2d.knn_nw_teacher_2d
# ──────────────────────────────────────────────────────────────────────────────

class TestKnnNWTeacher2D:
    def test_output_shape(self, small_sim):
        X_T, H, _, _ = small_sim
        query = X_T[:50]
        out = knn_nw_teacher_2d(X_T, H, query, k=30)
        assert out.shape == (50, 2)

    def test_no_nan(self, small_sim):
        X_T, H, _, _ = small_sim
        query = X_T[:50]
        out = knn_nw_teacher_2d(X_T, H, query, k=30)
        assert not torch.isnan(out).any()

    def test_k_larger_than_n_is_clamped(self, small_sim):
        """k > n should not raise — clamped internally."""
        X_T, H, _, _ = small_sim
        query = X_T[:10]
        out = knn_nw_teacher_2d(X_T, H, query, k=99_999)
        assert out.shape == (10, 2)
        assert not torch.isnan(out).any()


# ──────────────────────────────────────────────────────────────────────────────
# apply_teacher
# ──────────────────────────────────────────────────────────────────────────────

class TestApplyTeacher:
    @pytest.mark.parametrize("method", ["binned", "nw", "knn_nw", "raw"])
    def test_returns_triple(self, small_sim, method):
        X_T, H, _, _ = small_sim
        pts, sc, cc = apply_teacher(method, X_T, H, n_raw=200, knn_k=30, min_count=5)
        assert pts.ndim == 2 and pts.shape[1] == 2
        assert sc.shape == pts.shape
        assert cc.ndim == 1
        assert cc.shape[0] == pts.shape[0]

    @pytest.mark.parametrize("method", ["binned", "nw", "knn_nw", "raw"])
    def test_no_nan_in_output(self, small_sim, method):
        X_T, H, _, _ = small_sim
        pts, sc, cc = apply_teacher(method, X_T, H, n_raw=200, knn_k=30, min_count=5)
        assert not torch.isnan(pts).any()
        assert not torch.isnan(sc).any()
        assert not torch.isnan(cc).any()

    def test_raw_subsample_size(self, small_sim):
        X_T, H, _, _ = small_sim
        pts, sc, cc = apply_teacher("raw", X_T, H, n_raw=100)
        assert pts.shape[0] <= 100

    def test_unknown_method_raises(self, small_sim):
        X_T, H, _, _ = small_sim
        with pytest.raises(ValueError):
            apply_teacher("unknown", X_T, H)


# ──────────────────────────────────────────────────────────────────────────────
# build_all_datasets
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildAllDatasets:
    def test_keys_and_tensor_shapes(self):
        datasets = build_all_datasets(
            times=[0.20],
            device=DEVICE,
            n_raw=200,
            knn_k=30,
            n_bins=20,
            min_count=3,
            n_paths=500,
            n_steps=10,
        )
        for method in ("binned", "nw", "knn_nw", "raw"):
            assert method in datasets, f"missing {method}"
            t, x, s, c = datasets[method]
            assert t.ndim == 1
            assert x.shape == (t.shape[0], 2)
            assert s.shape == (t.shape[0], 2)
            assert c.shape == (t.shape[0],)

    def test_all_times_present(self):
        """Multiple times → all T values appear in t tensors."""
        times = [0.10, 0.30]
        datasets = build_all_datasets(
            times=times,
            device=DEVICE,
            n_raw=100,
            knn_k=20,
            n_bins=15,
            min_count=3,
            n_paths=500,
            n_steps=10,
        )
        t_vec = datasets["binned"][0]
        for T in times:
            assert (t_vec - T).abs().min().item() < 1e-6, f"T={T} not in dataset"


# ──────────────────────────────────────────────────────────────────────────────
# train_time_mlp (mini)
# ──────────────────────────────────────────────────────────────────────────────

class TestTrainTimeMlp:
    def test_model_output_shape(self):
        n = 200
        t = torch.rand(n, device=DEVICE)
        x = torch.randn(n, 2, device=DEVICE)
        s = torch.randn(n, 2, device=DEVICE)
        c = torch.ones(n, device=DEVICE)
        model = train_time_mlp(t, x, s, c, n_epochs=10, batch_size=64, device=DEVICE)
        with torch.no_grad():
            out = model(t, x)
        assert out.shape == (n, 2)

    def test_training_reduces_loss(self):
        """Loss after training should be less than initial (random) loss."""
        torch.manual_seed(0)
        n = 500
        t = torch.rand(n, device=DEVICE)
        x = torch.randn(n, 2, device=DEVICE)
        # deterministic targets: score(t, x) = -x (simple)
        s = -x.clone()
        c = torch.ones(n, device=DEVICE)

        from scoremodel_ext.malliavin.models import TimeScoreMLP2D
        model_init = TimeScoreMLP2D().to(DEVICE)
        with torch.no_grad():
            init_loss = ((model_init(t, x) - s) ** 2).mean().item()

        model = train_time_mlp(t, x, s, c, n_epochs=500, batch_size=64, device=DEVICE)
        with torch.no_grad():
            final_loss = ((model(t, x) - s) ** 2).mean().item()

        assert final_loss < init_loss, (
            f"Expected final_loss < init_loss, got {final_loss:.4f} vs {init_loss:.4f}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

class TestMetrics:
    @pytest.fixture
    def gmm_samples(self):
        """Perfect 8-GMM samples to test metrics near their ideal values."""
        rng = np.random.default_rng(0)
        radius = 2.0
        angles = np.linspace(0, 2 * math.pi, 9)[:-1]
        centers = radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)
        idx = rng.integers(0, 8, size=10_000)
        samples = centers[idx] + 0.08 * rng.standard_normal((10_000, 2))
        return samples, centers

    def test_mode_coverage_perfect(self, gmm_samples):
        samples, centers = gmm_samples
        mc = _mode_coverage(samples, centers)
        assert mc["n_covered"] == 8, f"Expected 8 covered modes, got {mc['n_covered']}"
        assert mc["coverage_fraction"] == 1.0

    def test_nearest_mode_dist_small_for_gmm(self, gmm_samples):
        samples, centers = gmm_samples
        nd = _nearest_mode_dist(samples, centers)
        assert nd["mean"] < 0.3, f"Expected mean < 0.3, got {nd['mean']:.4f}"

    def test_mmd_zero_for_same_distribution(self, gmm_samples):
        samples, _ = gmm_samples
        rng = np.random.default_rng(1)
        mmd = _mmd_rbf(samples, samples, n_sub=500, rng=rng)
        # Unbiased MMD^2 for same sample set should be ~0 (can be slightly negative)
        assert abs(mmd) < 0.1, f"Expected MMD~0 for same data, got {mmd:.4f}"

    def test_mmd_positive_for_different_distributions(self, gmm_samples):
        samples, _ = gmm_samples
        noise = np.random.default_rng(2).standard_normal(samples.shape) * 3.0
        rng = np.random.default_rng(3)
        mmd = _mmd_rbf(samples, noise, n_sub=500, rng=rng)
        assert mmd > 0.0, f"Expected positive MMD for different distributions, got {mmd:.4f}"

    def test_sliced_wasserstein_zero_for_same(self, gmm_samples):
        samples, _ = gmm_samples
        rng = np.random.default_rng(4)
        sw = _sliced_wasserstein(samples, samples, rng=rng)
        assert sw < 0.01, f"Expected SW~0 for same data, got {sw:.4f}"

    def test_sliced_wasserstein_positive_for_different(self, gmm_samples):
        samples, _ = gmm_samples
        shifted = samples + 5.0
        rng = np.random.default_rng(5)
        sw = _sliced_wasserstein(samples, shifted, rng=rng)
        assert sw > 1.0, f"Expected large SW for shifted data, got {sw:.4f}"

    def test_compute_metrics_keys(self, gmm_samples):
        samples, centers = gmm_samples
        m = compute_metrics(samples, centers, nan_rate=0.01)
        for key in ("mode_coverage", "nearest_mode_dist", "mmd_rbf",
                    "sliced_wasserstein", "nan_rate", "n_samples"):
            assert key in m, f"Missing key {key!r} in metrics"

    def test_compute_metrics_nan_rate_passthrough(self, gmm_samples):
        samples, centers = gmm_samples
        m = compute_metrics(samples, centers, nan_rate=0.05)
        assert abs(m["nan_rate"] - 0.05) < 1e-9


# ──────────────────────────────────────────────────────────────────────────────
# knn_nw_teacher_2d — bandwidth_scale
# ──────────────────────────────────────────────────────────────────────────────

class TestKnnNWBandwidthScale:
    def test_scale_changes_output(self, small_sim):
        """Different bandwidth_scale values should produce different scores."""
        X_T, H, _, _ = small_sim
        query = X_T[:40]
        out1 = knn_nw_teacher_2d(X_T, H, query, k=30, bandwidth_scale=0.5)
        out2 = knn_nw_teacher_2d(X_T, H, query, k=30, bandwidth_scale=1.5)
        assert not torch.allclose(out1, out2, atol=1e-4), (
            "bandwidth_scale=0.5 and 1.5 should give different results"
        )

    def test_scale_one_matches_default(self, small_sim):
        """bandwidth_scale=1.0 should match the original default behaviour."""
        X_T, H, _, _ = small_sim
        query = X_T[:20]
        out_default = knn_nw_teacher_2d(X_T, H, query, k=30)
        out_scale1  = knn_nw_teacher_2d(X_T, H, query, k=30, bandwidth_scale=1.0)
        assert torch.allclose(out_default, out_scale1, atol=1e-6)

    def test_no_nan_with_various_scales(self, small_sim):
        X_T, H, _, _ = small_sim
        query = X_T[:30]
        for scale in (0.5, 0.75, 1.0, 2.0):
            out = knn_nw_teacher_2d(X_T, H, query, k=30, bandwidth_scale=scale)
            assert not torch.isnan(out).any(), f"NaN at scale={scale}"


# ──────────────────────────────────────────────────────────────────────────────
# apply_teacher_sweep
# ──────────────────────────────────────────────────────────────────────────────

class TestApplyTeacherSweep:
    @pytest.mark.parametrize("family,kwargs", [
        ("nw",     {"bandwidth": 0.20}),
        ("nw",     {"bandwidth": None}),   # Silverman auto
        ("knn_nw", {"k": 30, "bandwidth_scale": 0.75}),
        ("knn_nw", {"k": 30, "bandwidth_scale": 1.0}),
    ])
    def test_returns_triple(self, small_sim, family, kwargs):
        X_T, H, _, _ = small_sim
        pts, sc, cc = apply_teacher_sweep(family, X_T, H, min_count=5, **kwargs)
        assert pts.ndim == 2 and pts.shape[1] == 2
        assert sc.shape == pts.shape
        assert cc.shape == (pts.shape[0],)

    def test_no_nan(self, small_sim):
        X_T, H, _, _ = small_sim
        for family, kw in [
            ("nw", {"bandwidth": 0.15}),
            ("knn_nw", {"k": 30, "bandwidth_scale": 0.5}),
        ]:
            pts, sc, cc = apply_teacher_sweep(family, X_T, H, min_count=5, **kw)
            assert not torch.isnan(sc).any(), f"NaN in {family} sweep teacher"

    def test_unknown_family_raises(self, small_sim):
        X_T, H, _, _ = small_sim
        with pytest.raises(ValueError):
            apply_teacher_sweep("unknown", X_T, H, min_count=5)

    def test_nw_fixed_vs_auto_bandwidth_differ(self, small_sim):
        """Fixed bandwidth should generally differ from Silverman auto."""
        X_T, H, _, _ = small_sim
        # use small n_bins/min_count so bins are non-empty with n=1000
        pts1, sc1, _ = apply_teacher_sweep("nw", X_T, H, bandwidth=0.05, n_bins=15, min_count=3)
        pts2, sc2, _ = apply_teacher_sweep("nw", X_T, H, bandwidth=None, n_bins=15, min_count=3)
        # Only assert when both have non-empty outputs
        if pts1.shape[0] > 0 and pts1.shape == pts2.shape:
            assert not torch.allclose(sc1, sc2, atol=1e-3), (
                "Fixed h=0.05 and Silverman bandwidth gave identical scores — unexpected"
            )


# ──────────────────────────────────────────────────────────────────────────────
# simulate_all_times
# ──────────────────────────────────────────────────────────────────────────────

class TestSimulateAllTimes:
    def test_length_matches_times(self):
        times = [0.10, 0.30]
        cache = simulate_all_times(
            times, DEVICE,
            n_paths=300, n_steps=10,
        )
        assert len(cache) == len(times)

    def test_tuple_structure(self):
        cache = simulate_all_times(
            [0.20], DEVICE,
            n_paths=300, n_steps=10,
        )
        T, X_T, H = cache[0]
        assert abs(T - 0.20) < 1e-9
        assert X_T.shape == (300, 2)
        assert H.shape == (300, 2)

    def test_T_values_correct(self):
        times = [0.15, 0.40]
        cache = simulate_all_times(
            times, DEVICE,
            n_paths=300, n_steps=10,
        )
        for (T, _, _), expected in zip(cache, times):
            assert abs(T - expected) < 1e-9


# ──────────────────────────────────────────────────────────────────────────────
# build_sweep_dataset_from_cache
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildSweepDatasetFromCache:
    @pytest.fixture(scope="class")
    def tiny_cache(self):
        return simulate_all_times(
            [0.10, 0.30], DEVICE,
            n_paths=500, n_steps=10,
        )

    @pytest.mark.parametrize("family,kwargs", [
        ("nw",     {"bandwidth": 0.20}),
        ("knn_nw", {"k": 20, "bandwidth_scale": 1.0}),
    ])
    def test_returns_tensors(self, tiny_cache, family, kwargs):
        result = build_sweep_dataset_from_cache(
            tiny_cache, family, min_count=3, **kwargs
        )
        assert result is not None
        t, x, s, c = result
        assert t.ndim == 1
        assert x.shape == (t.shape[0], 2)
        assert s.shape == (t.shape[0], 2)
        assert c.shape == (t.shape[0],)

    def test_multiple_times_concatenated(self, tiny_cache):
        result = build_sweep_dataset_from_cache(
            tiny_cache, "nw", bandwidth=0.20, min_count=3
        )
        assert result is not None
        t, _, _, _ = result
        # t should contain both 0.10 and 0.30 values
        assert (t - 0.10).abs().min().item() < 1e-6
        assert (t - 0.30).abs().min().item() < 1e-6

    def test_no_nan_in_dataset(self, tiny_cache):
        result = build_sweep_dataset_from_cache(
            tiny_cache, "knn_nw", k=20, bandwidth_scale=0.75, min_count=3
        )
        assert result is not None
        t, x, s, c = result
        for name, tensor in [("t", t), ("x", x), ("s", s), ("c", c)]:
            assert not torch.isnan(tensor).any(), f"NaN in {name}"


# ──────────────────────────────────────────────────────────────────────────────
# _aggregate_seeds
# ──────────────────────────────────────────────────────────────────────────────

class TestAggregateSeeds:
    def _make_metrics(self, mmd, sw):
        return {
            "mmd_rbf": mmd,
            "sliced_wasserstein": sw,
            "mode_coverage": {"coverage_fraction": 1.0},
            "nearest_mode_dist": {"mean": 0.1},
            "nan_rate": 0.0,
            "n_samples": 1000,
        }

    def test_single_seed_std_zero(self):
        agg = _aggregate_seeds([self._make_metrics(0.01, 0.05)])
        assert agg["mmd_rbf_mean"] == pytest.approx(0.01)
        assert agg["mmd_rbf_std"]  == pytest.approx(0.0)
        assert agg["sliced_wasserstein_mean"] == pytest.approx(0.05)
        assert agg["sliced_wasserstein_std"]  == pytest.approx(0.0)

    def test_mean_correct(self):
        seeds = [self._make_metrics(mmd, sw)
                 for mmd, sw in [(0.01, 0.10), (0.03, 0.20), (0.02, 0.15)]]
        agg = _aggregate_seeds(seeds)
        assert agg["mmd_rbf_mean"] == pytest.approx(0.02, abs=1e-9)
        assert agg["sliced_wasserstein_mean"] == pytest.approx(0.15, abs=1e-9)

    def test_std_correct(self):
        import statistics
        mmds = [0.01, 0.02, 0.03]
        seeds = [self._make_metrics(m, 0.0) for m in mmds]
        agg = _aggregate_seeds(seeds)
        expected_std = statistics.stdev(mmds)
        assert agg["mmd_rbf_std"] == pytest.approx(expected_std, rel=1e-5)

    def test_output_keys(self):
        agg = _aggregate_seeds([self._make_metrics(0.01, 0.05)])
        for key in ("mmd_rbf_mean", "mmd_rbf_std",
                    "sliced_wasserstein_mean", "sliced_wasserstein_std"):
            assert key in agg, f"Missing key: {key}"


# ──────────────────────────────────────────────────────────────────────────────
# _run_single_seed  (minimal smoke test — tiny n_paths, epochs)
# ──────────────────────────────────────────────────────────────────────────────

class TestRunSingleSeed:
    _MINI_KW = dict(
        n_paths=300,
        n_steps=10,
        n_epochs=20,
        batch_size=64,
        n_samples_reverse=200,
        n_steps_reverse=20,
    )

    def test_returns_metric_keys(self, tmp_path):
        cfg = {"family": "nw", "bandwidth": 0.20, "k": None, "bandwidth_scale": 1.0}
        m = _run_single_seed(
            config=cfg,
            times=[0.20, 0.40],
            seed=0,
            outdir=tmp_path / "seed0",
            device=DEVICE,
            sigma_min=0.15, sigma_max=1.20, gamma_reg=1e-3,
            n_bins=15, min_count=3, lr=2e-4,
            **self._MINI_KW,
        )
        assert m is not None
        for key in ("mmd_rbf", "sliced_wasserstein", "mode_coverage", "nan_rate"):
            assert key in m, f"Missing key: {key}"

    def test_seed_reproducibility(self, tmp_path):
        """Two runs with the same seed should give identical MMD."""
        cfg = {"family": "knn_nw", "bandwidth": None, "k": 20, "bandwidth_scale": 1.0}
        kw = dict(
            config=cfg, times=[0.20], seed=7,
            device=DEVICE, sigma_min=0.15, sigma_max=1.20, gamma_reg=1e-3,
            n_bins=15, min_count=3, lr=2e-4, **self._MINI_KW,
        )
        m1 = _run_single_seed(outdir=tmp_path / "run1", **kw)
        m2 = _run_single_seed(outdir=tmp_path / "run2", **kw)
        assert m1 is not None and m2 is not None
        assert m1["mmd_rbf"] == pytest.approx(m2["mmd_rbf"], rel=1e-4), (
            "Same seed should reproduce same MMD"
        )

    def test_different_seeds_may_differ(self, tmp_path):
        """Different seeds can produce different MMD (probabilistic — just runs without error)."""
        cfg = {"family": "nw", "bandwidth": 0.15, "k": None, "bandwidth_scale": 1.0}
        base_kw = dict(
            config=cfg, times=[0.20], device=DEVICE,
            sigma_min=0.15, sigma_max=1.20, gamma_reg=1e-3,
            n_bins=15, min_count=3, lr=2e-4, **self._MINI_KW,
        )
        m0 = _run_single_seed(seed=0, outdir=tmp_path / "s0", **base_kw)
        m1 = _run_single_seed(seed=1, outdir=tmp_path / "s1", **base_kw)
        # Both should complete without error
        assert m0 is not None and m1 is not None

    def test_output_files_saved(self, tmp_path):
        cfg = {"family": "nw", "bandwidth": 0.20, "k": None, "bandwidth_scale": 1.0}
        outdir = tmp_path / "seed42"
        _run_single_seed(
            config=cfg, times=[0.20], seed=42, outdir=outdir,
            device=DEVICE, sigma_min=0.15, sigma_max=1.20, gamma_reg=1e-3,
            n_bins=15, min_count=3, lr=2e-4, **self._MINI_KW,
        )
        assert (outdir / "metrics_seed42.json").exists()
        assert (outdir / "reverse_samples_seed42.png").exists()


# ──────────────────────────────────────────────────────────────────────────────
# _build_baseline_dataset_from_cache
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildBaselineDatasetFromCache:
    """Unit tests for the baseline dataset builder (raw / binned)."""

    @pytest.fixture(scope="class")
    def sim_cache_small(self):
        cache = simulate_all_times(
            [0.10, 0.30],
            device=DEVICE,
            n_paths=400,
            n_steps=10,
            sigma_min=0.15,
            sigma_max=1.20,
            gamma_reg=1e-3,
        )
        return cache

    def test_raw_returns_tensors(self, sim_cache_small):
        result = _build_baseline_dataset_from_cache(
            sim_cache_small, "raw", n_bins=15, min_count=2
        )
        assert result is not None, "raw should return data"
        t, x, s, c = result
        assert t.ndim == 1
        assert x.shape[1] == 2
        assert s.shape == x.shape

    def test_binned_returns_tensors(self, sim_cache_small):
        result = _build_baseline_dataset_from_cache(
            sim_cache_small, "binned", n_bins=15, min_count=2
        )
        assert result is not None, "binned should return data"
        t, x, s, c = result
        assert t.ndim == 1
        assert x.shape[1] == 2

    def test_lengths_consistent(self, sim_cache_small):
        t, x, s, c = _build_baseline_dataset_from_cache(
            sim_cache_small, "binned", n_bins=15, min_count=2
        )
        assert t.shape[0] == x.shape[0] == s.shape[0] == c.shape[0]

    def test_invalid_method_raises(self, sim_cache_small):
        with pytest.raises(ValueError):
            _build_baseline_dataset_from_cache(sim_cache_small, "knn_nw")


# ──────────────────────────────────────────────────────────────────────────────
# _run_single_seed  —  baseline code paths (raw / binned)
# ──────────────────────────────────────────────────────────────────────────────

class TestRunSingleSeedBaseline:
    _MINI_KW = dict(
        n_paths=300,
        n_steps=10,
        n_epochs=20,
        batch_size=64,
        n_samples_reverse=200,
        n_steps_reverse=20,
    )

    @pytest.mark.parametrize("method", ["raw", "binned"])
    def test_baseline_returns_metric_keys(self, tmp_path, method):
        cfg = {"method": method}
        m = _run_single_seed(
            config=cfg,
            times=[0.20, 0.40],
            seed=0,
            outdir=tmp_path / method / "seed0",
            device=DEVICE,
            sigma_min=0.15, sigma_max=1.20, gamma_reg=1e-3,
            n_bins=15, min_count=3, lr=2e-4,
            **self._MINI_KW,
        )
        assert m is not None
        for key in ("mmd_rbf", "sliced_wasserstein", "mode_coverage", "nan_rate"):
            assert key in m, f"Missing key {key!r} for method={method}"

    @pytest.mark.parametrize("method", ["raw", "binned"])
    def test_baseline_output_files_saved(self, tmp_path, method):
        cfg = {"method": method}
        outdir = tmp_path / method / "seed0"
        _run_single_seed(
            config=cfg, times=[0.20], seed=0, outdir=outdir,
            device=DEVICE, sigma_min=0.15, sigma_max=1.20, gamma_reg=1e-3,
            n_bins=15, min_count=3, lr=2e-4, **self._MINI_KW,
        )
        assert (outdir / "metrics_seed0.json").exists()
        assert (outdir / "reverse_samples_seed0.png").exists()


# ──────────────────────────────────────────────────────────────────────────────
# _BASELINE_CONFIGS constant
# ──────────────────────────────────────────────────────────────────────────────

class TestBaselineConfigs:
    def test_both_baselines_present(self):
        assert "baseline_raw" in _BASELINE_CONFIGS
        assert "baseline_binned" in _BASELINE_CONFIGS

    def test_raw_config_has_method_key(self):
        assert _BASELINE_CONFIGS["baseline_raw"].get("method") == "raw"

    def test_binned_config_has_method_key(self):
        assert _BASELINE_CONFIGS["baseline_binned"].get("method") == "binned"

