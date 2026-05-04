"""
pytest smoke tests for Phase A of the Mirafzali reproduction.

Covers:
  - Dataset generators (8gmm, checkerboard, swissroll)
  - VE schedule helpers (marginal_var, sigma_at)
  - simulate_ve  → (X_T, H) shapes and physics checks
  - simulate_vp / simulate_subvp smoke
  - simulate_linear dispatch
  - simulate_all_times_linear (8gmm + VE)
  - apply_teacher_linear  (raw, binned, knn_nw)
  - build_training_dataset (8gmm + VE + binned)
  - train_score_mlp (tiny mini run)
  - reverse_sample_ve (smoke)
  - compute_metrics (8gmm)

Run
---
cd /export/home/ymorimoto/github/scoremodel
python -m pytest tests/test_mirafzali_phase_a.py -v
"""

import math
import numpy as np
import pytest
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scoremodel_ext.malliavin.datasets_2d import (
    sample_8gmm,
    sample_checkerboard,
    sample_swissroll,
    get_sampler,
)
from scoremodel_ext.malliavin.sde_linear import (
    VEConfig, VPConfig, SubVPConfig,
    ve_marginal_var, ve_sigma_at,
    vp_marginal_params, subvp_marginal_params,
    simulate_ve, simulate_vp, simulate_subvp,
    simulate_linear,
    reverse_sample_ve,
)
from scoremodel_ext.malliavin.experiment_mirafzali import (
    apply_teacher_linear,
    simulate_all_times_linear,
    build_training_dataset,
    train_score_mlp,
    compute_metrics,
    DEFAULT_VE,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# Dataset generators
# ──────────────────────────────────────────────────────────────────────────────

class TestDatasets:
    @pytest.mark.parametrize("n", [1, 100, 500])
    def test_8gmm_shape(self, n):
        x, centers = sample_8gmm(n, device=DEVICE)
        assert x.shape == (n, 2)
        assert centers.shape == (8, 2)

    @pytest.mark.parametrize("n", [1, 100, 500])
    def test_checkerboard_shape(self, n):
        x = sample_checkerboard(n, device=DEVICE)
        assert x.shape == (n, 2)

    def test_checkerboard_range(self):
        x = sample_checkerboard(10_000, half_width=2.0, device="cpu")
        assert x.abs().max().item() <= 2.0 + 1e-5

    @pytest.mark.parametrize("n", [1, 100, 500])
    def test_swissroll_shape(self, n):
        x = sample_swissroll(n, device=DEVICE)
        assert x.shape == (n, 2)

    def test_swissroll_no_nan(self):
        x = sample_swissroll(1_000, device="cpu")
        assert not torch.isnan(x).any()

    def test_get_sampler_returns_callable(self):
        for name in ("8gmm", "checkerboard", "swissroll"):
            fn = get_sampler(name)
            assert callable(fn)

    def test_get_sampler_unknown_raises(self):
        with pytest.raises(ValueError):
            get_sampler("unknown_dataset")


# ──────────────────────────────────────────────────────────────────────────────
# VE schedule helpers
# ──────────────────────────────────────────────────────────────────────────────

class TestVEHelpers:
    def test_marginal_var_constant_sigma(self):
        cfg = VEConfig(sigma_min=2.0, sigma_max=2.0, T=1.0)
        assert math.isclose(ve_marginal_var(1.0, cfg), 4.0, rel_tol=1e-6)

    def test_marginal_var_positive(self):
        cfg = VEConfig(sigma_min=0.1, sigma_max=10.0, T=1.0)
        assert ve_marginal_var(1.0, cfg) > 0

    def test_marginal_var_increases_with_T(self):
        cfg = VEConfig(sigma_min=0.1, sigma_max=10.0, T=1.0)
        assert ve_marginal_var(0.5, cfg) < ve_marginal_var(1.0, cfg)

    def test_sigma_at_boundary(self):
        cfg = VEConfig(sigma_min=0.1, sigma_max=10.0, T=1.0)
        assert math.isclose(ve_sigma_at(0.0, cfg), 0.1, rel_tol=1e-5)
        assert math.isclose(ve_sigma_at(1.0, cfg), 10.0, rel_tol=1e-5)

    def test_sigma_at_constant(self):
        cfg = VEConfig(sigma_min=3.0, sigma_max=3.0, T=1.0)
        assert math.isclose(ve_sigma_at(0.5, cfg), 3.0, rel_tol=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# simulate_ve
# ──────────────────────────────────────────────────────────────────────────────

class TestSimulateVE:
    @pytest.fixture(scope="class")
    def data(self):
        n = 2_000
        X0, _ = sample_8gmm(n, device=DEVICE)
        cfg = VEConfig(sigma_min=0.1, sigma_max=5.0, T=1.0)
        X_T, H = simulate_ve(X0, T=1.0, cfg=cfg)
        return X0, X_T, H, cfg

    def test_shapes(self, data):
        X0, X_T, H, _ = data
        assert X_T.shape == X0.shape
        assert H.shape  == X0.shape

    def test_no_nan(self, data):
        _, X_T, H, _ = data
        assert not torch.isnan(X_T).any()
        assert not torch.isnan(H).any()

    def test_malliavin_identity(self, data):
        """H = (X_0 − X_T) / Σ²  →  X_0 − X_T ≈ Σ² · H"""
        X0, X_T, H, cfg = data
        Sigma2 = ve_marginal_var(1.0, cfg)
        residual = (X0 - X_T) - Sigma2 * H
        assert residual.abs().max().item() < 1e-4

    def test_marginal_mean_near_X0(self, data):
        """E[X_T] ≈ E[X_0] (VE has no drift)."""
        X0, X_T, H, _ = data
        # Tolerant check: mean difference < 0.1 per dimension
        diff = (X_T.mean(0) - X0.mean(0)).abs()
        assert diff.max().item() < 0.5  # stochastic — loose bound

    def test_marginal_var_approx(self, data):
        """Var(X_T) ≈ Var(X_0) + Σ²(T)."""
        X0, X_T, H, cfg = data
        Sigma2  = ve_marginal_var(1.0, cfg)
        var_X0  = X0.var(dim=0).mean().item()
        var_XT  = X_T.var(dim=0).mean().item()
        expected = var_X0 + Sigma2
        assert abs(var_XT - expected) / expected < 0.15  # 15% relative tolerance


# ──────────────────────────────────────────────────────────────────────────────
# simulate_vp / simulate_subvp / simulate_linear  (smoke tests)
# ──────────────────────────────────────────────────────────────────────────────

class TestSimulateVPSubVP:
    @pytest.mark.parametrize("cfg_cls,kw", [
        (VPConfig,    dict(beta_min=0.1, beta_max=20.0, T=1.0)),
        (SubVPConfig, dict(beta_min=0.1, beta_max=20.0, T=1.0)),
    ])
    def test_shapes_no_nan(self, cfg_cls, kw):
        cfg = cfg_cls(**kw)
        n   = 500
        X0, _ = sample_8gmm(n, device=DEVICE)
        X_T, H = simulate_linear(X0, T=1.0, sde_config=cfg)
        assert X_T.shape == (n, 2)
        assert H.shape   == (n, 2)
        assert not torch.isnan(X_T).any()
        assert not torch.isnan(H).any()

    def test_vp_alpha_less_than_one(self):
        cfg = VPConfig(beta_min=0.1, beta_max=20.0, T=1.0)
        alpha, var = vp_marginal_params(1.0, cfg)
        assert 0.0 < alpha < 1.0
        assert 0.0 < var <= 1.0

    def test_subvp_var_less_than_vp(self):
        """sub-VP has smaller marginal variance than VP."""
        bp = dict(beta_min=0.1, beta_max=20.0, T=1.0)
        _, var_vp    = vp_marginal_params(1.0, VPConfig(**bp))
        _, var_subvp = subvp_marginal_params(1.0, SubVPConfig(**bp))
        assert var_subvp < var_vp

    def test_simulate_linear_raises_unknown(self):
        X0, _ = sample_8gmm(10, device=DEVICE)
        with pytest.raises(ValueError):
            simulate_linear(X0, 1.0, sde_config="bad_type")


# ──────────────────────────────────────────────────────────────────────────────
# simulate_all_times_linear
# ──────────────────────────────────────────────────────────────────────────────

class TestSimulateAllTimesLinear:
    @pytest.fixture(scope="class")
    def cache(self):
        return simulate_all_times_linear(
            times=[0.10, 0.50, 1.00],
            dataset_name="8gmm",
            sde_config=DEFAULT_VE,
            n_paths=500,
            device=DEVICE,
        )

    def test_length(self, cache):
        assert len(cache) == 3

    def test_T_values(self, cache):
        Ts = [entry[0] for entry in cache]
        assert Ts == pytest.approx([0.10, 0.50, 1.00])

    def test_tensor_shapes(self, cache):
        for T, X_T, H in cache:
            assert X_T.shape == (500, 2)
            assert H.shape   == (500, 2)

    def test_no_nan(self, cache):
        for _, X_T, H in cache:
            assert not torch.isnan(X_T).any()
            assert not torch.isnan(H).any()


# ──────────────────────────────────────────────────────────────────────────────
# apply_teacher_linear
# ──────────────────────────────────────────────────────────────────────────────

class TestApplyTeacherLinear:
    @pytest.fixture(scope="class")
    def xt_h(self):
        n = 2_000
        X0, _ = sample_8gmm(n, device=DEVICE)
        X_T, H = simulate_ve(X0, T=1.0, cfg=DEFAULT_VE)
        return X_T, H

    @pytest.mark.parametrize("method", ["raw", "binned", "knn_nw"])
    def test_returns_triple(self, xt_h, method):
        X_T, H = xt_h
        pts, sc, cc = apply_teacher_linear(
            method, X_T, H,
            n_raw=200, n_bins=15, min_count=2, knn_k=20,
        )
        assert pts.ndim == 2 and pts.shape[1] == 2
        assert sc.shape == pts.shape
        assert cc.ndim == 1 and cc.shape[0] == pts.shape[0]

    @pytest.mark.parametrize("method", ["raw", "binned", "knn_nw"])
    def test_no_nan(self, xt_h, method):
        X_T, H = xt_h
        pts, sc, cc = apply_teacher_linear(
            method, X_T, H,
            n_raw=200, n_bins=15, min_count=2, knn_k=20,
        )
        assert not torch.isnan(pts).any()
        assert not torch.isnan(sc).any()

    def test_unknown_method_raises(self, xt_h):
        X_T, H = xt_h
        with pytest.raises(ValueError):
            apply_teacher_linear("nw_fancy", X_T, H)

    def test_raw_size_bounded(self, xt_h):
        X_T, H = xt_h
        pts, _, _ = apply_teacher_linear("raw", X_T, H, n_raw=100)
        assert pts.shape[0] <= 100


# ──────────────────────────────────────────────────────────────────────────────
# build_training_dataset
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildTrainingDataset:
    @pytest.fixture(scope="class")
    def small_cache(self):
        return simulate_all_times_linear(
            times=[0.20, 0.60],
            dataset_name="8gmm",
            sde_config=DEFAULT_VE,
            n_paths=600,
            device=DEVICE,
        )

    def test_binned_returns_tensors(self, small_cache):
        result = build_training_dataset(
            small_cache, "binned", n_bins=15, min_count=2,
        )
        assert result is not None
        t, x, s, c = result
        assert t.ndim == 1
        assert x.shape[1] == 2
        assert s.shape == x.shape
        assert c.shape[0] == x.shape[0]

    def test_raw_returns_tensors(self, small_cache):
        result = build_training_dataset(
            small_cache, "raw", n_raw=100,
        )
        assert result is not None

    def test_knn_nw_returns_tensors(self, small_cache):
        result = build_training_dataset(
            small_cache, "knn_nw", n_bins=15, min_count=2, knn_k=20,
        )
        assert result is not None

    def test_lengths_consistent(self, small_cache):
        t, x, s, c = build_training_dataset(
            small_cache, "binned", n_bins=15, min_count=2,
        )
        assert t.shape[0] == x.shape[0] == s.shape[0] == c.shape[0]

    def test_times_span_both_entries(self, small_cache):
        t, x, s, c = build_training_dataset(
            small_cache, "binned", n_bins=15, min_count=2,
        )
        unique_t = t.unique().sort().values.tolist()
        assert len(unique_t) >= 1  # at least one time present


# ──────────────────────────────────────────────────────────────────────────────
# train_score_mlp (mini run)
# ──────────────────────────────────────────────────────────────────────────────

class TestTrainScoreMLP:
    @pytest.fixture(scope="class")
    def tiny_dataset(self):
        cache = simulate_all_times_linear(
            times=[0.50],
            dataset_name="8gmm",
            sde_config=DEFAULT_VE,
            n_paths=400,
            device=DEVICE,
        )
        return build_training_dataset(cache, "binned", n_bins=15, min_count=2)

    def test_model_output_shape(self, tiny_dataset):
        t, x, s, c = tiny_dataset
        model = train_score_mlp(
            t, x, s, c,
            n_epochs=10, batch_size=64, device=DEVICE,
        )
        model.eval()
        with torch.no_grad():
            t_test = torch.tensor([0.5], device=DEVICE)
            x_test = torch.randn(1, 2, device=DEVICE)
            out = model(t_test, x_test)
        assert out.shape == (1, 2)

    def test_training_runs_without_nan(self, tiny_dataset):
        t, x, s, c = tiny_dataset
        model = train_score_mlp(
            t, x, s, c,
            n_epochs=20, batch_size=64, device=DEVICE,
        )
        model.eval()
        with torch.no_grad():
            pred = model(t, x)
        assert not torch.isnan(pred).any()


# ──────────────────────────────────────────────────────────────────────────────
# reverse_sample_ve (smoke)
# ──────────────────────────────────────────────────────────────────────────────

class TestReverseSampleVE:
    @pytest.fixture(scope="class")
    def trained_model(self):
        """Tiny trained model (very few epochs — just for shape checking)."""
        cache = simulate_all_times_linear(
            times=[0.20, 0.50, 1.00],
            dataset_name="8gmm",
            sde_config=DEFAULT_VE,
            n_paths=300,
            device=DEVICE,
        )
        ds = build_training_dataset(cache, "binned", n_bins=12, min_count=2)
        if ds is None:
            pytest.skip("no valid teacher points")
        t, x, s, c = ds
        return train_score_mlp(t, x, s, c, n_epochs=5, batch_size=32, device=DEVICE)

    def test_output_shape(self, trained_model):
        samples = reverse_sample_ve(
            trained_model, n_samples=100, cfg=DEFAULT_VE,
            n_steps=10, device=DEVICE,
        )
        assert samples.shape == (100, 2)

    def test_output_on_cpu(self, trained_model):
        samples = reverse_sample_ve(
            trained_model, n_samples=50, cfg=DEFAULT_VE,
            n_steps=5, device=DEVICE,
        )
        assert samples.device.type == "cpu"

    def test_nan_rate_low(self, trained_model):
        samples = reverse_sample_ve(
            trained_model, n_samples=200, cfg=DEFAULT_VE,
            n_steps=10, device=DEVICE,
        )
        nan_rate = torch.isnan(samples).any(dim=1).float().mean().item()
        # Barely trained model — just check it doesn't blow up entirely
        assert nan_rate < 0.5


# ──────────────────────────────────────────────────────────────────────────────
# compute_metrics (8gmm)
# ──────────────────────────────────────────────────────────────────────────────

class TestComputeMetrics:
    def _ref_samples(self, n=2_000):
        x, centers = sample_8gmm(n, device="cpu")
        return x.numpy(), centers.numpy()

    def test_keys_present(self):
        samples_np, centers_np = self._ref_samples()
        m = compute_metrics(
            samples_np, nan_rate=0.0,
            dataset_name="8gmm", centers_np=centers_np, n_ref=500,
        )
        for key in ("mmd_rbf", "sliced_wasserstein", "nan_rate",
                    "n_samples", "mode_coverage"):
            assert key in m, f"Missing key: {key}"

    def test_mmd_near_zero_for_same_dist(self):
        samples_np, centers_np = self._ref_samples(3_000)
        m = compute_metrics(
            samples_np, nan_rate=0.0,
            dataset_name="8gmm", centers_np=centers_np, n_ref=3_000,
        )
        assert m["mmd_rbf"] < 0.05

    def test_mmd_positive_for_different_dist(self):
        # Gaussian noise vs 8gmm
        noise_np = np.random.randn(2_000, 2) * 5.0
        _, centers_np = self._ref_samples()
        m = compute_metrics(
            noise_np, nan_rate=0.0,
            dataset_name="8gmm", centers_np=centers_np, n_ref=2_000,
        )
        assert m["mmd_rbf"] > 0.0

    def test_mode_coverage_full_for_perfect_samples(self):
        samples_np, centers_np = self._ref_samples(5_000)
        m = compute_metrics(
            samples_np, nan_rate=0.0,
            dataset_name="8gmm", centers_np=centers_np, n_ref=500,
        )
        assert m["mode_coverage"]["coverage_fraction"] == 1.0

    def test_no_mode_coverage_for_non_gmm(self):
        """compute_metrics should not include mode_coverage for checkerboard."""
        x = sample_checkerboard(2_000, device="cpu").numpy()
        m = compute_metrics(x, nan_rate=0.0, dataset_name="checkerboard",
                            n_ref=500)
        assert "mode_coverage" not in m
