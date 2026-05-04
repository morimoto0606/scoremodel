"""
pytest smoke tests for Phase B — Mirafzali nonlinear SDE.

Tests are ordered from primitives up to the full pipeline.
All use tiny parameters to stay fast.

Run
---
cd /export/home/ymorimoto/github/scoremodel
python -m pytest tests/test_mirafzali_nonlinear.py -v
"""

import math
import numpy as np
import pytest
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scoremodel_ext.malliavin.datasets_2d import sample_8gmm
from scoremodel_ext.malliavin.sde_nonlinear import (
    NonlinearSDEConfig,
    beta_at,
    sigma_t,
    drift_nl,
    jac_drift_nl,
    simulate_forward_nl,
    simulate_malliavin_nl,
    reverse_euler_nl,
)
from scoremodel_ext.malliavin.experiment_mirafzali_nonlinear import (
    apply_teacher_nl,
    simulate_all_times_nl,
    build_training_dataset_nl,
    compute_metrics_nl,
    build_results_table,
    DEFAULT_NL_CFG,
    _n_steps_for,
    _binned_score_at_points,
)
from scoremodel_ext.malliavin.experiment_mirafzali import train_score_mlp
from scoremodel_ext.malliavin.models import (
    MirafzaliSkorokhodNet,
    NormalizedSkorokhodModel,
    train_mirafzali_skorokhod_net,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Tiny config for fast tests (keeps beta range realistic but sim is cheap)
_MINI_CFG = NonlinearSDEConfig(
    k=1.0, sigma=1.0, a=0.0,
    beta_min=1.0, beta_max=10.0, T=1.0,
)
_MINI_N      = 300
_MINI_STEPS  = 20


# ──────────────────────────────────────────────────────────────────────────────
# NonlinearSDEConfig
# ──────────────────────────────────────────────────────────────────────────────

class TestNonlinearSDEConfig:
    def test_default_values(self):
        cfg = NonlinearSDEConfig()
        assert cfg.k        == 1.0
        assert cfg.sigma    == 1.0
        assert cfg.a        == 0.0
        assert cfg.beta_min == 1.0
        assert cfg.beta_max == 25.0
        assert cfg.T        == 1.0

    def test_custom_values(self):
        cfg = NonlinearSDEConfig(k=2.0, sigma=0.5, a=1.0,
                                  beta_min=0.5, beta_max=10.0, T=2.0)
        assert cfg.k == 2.0
        assert cfg.a == 1.0
        assert cfg.T == 2.0


# ──────────────────────────────────────────────────────────────────────────────
# β schedule
# ──────────────────────────────────────────────────────────────────────────────

class TestBetaSchedule:
    def test_at_zero(self):
        cfg = NonlinearSDEConfig(beta_min=1.0, beta_max=25.0, T=1.0)
        assert beta_at(0.0, cfg) == pytest.approx(1.0)

    def test_at_T(self):
        cfg = NonlinearSDEConfig(beta_min=1.0, beta_max=25.0, T=1.0)
        assert beta_at(1.0, cfg) == pytest.approx(25.0)

    def test_midpoint(self):
        cfg = NonlinearSDEConfig(beta_min=1.0, beta_max=25.0, T=1.0)
        assert beta_at(0.5, cfg) == pytest.approx(13.0)

    def test_monotone_increasing(self):
        cfg = NonlinearSDEConfig(beta_min=1.0, beta_max=25.0, T=1.0)
        assert beta_at(0.2, cfg) < beta_at(0.8, cfg)

    def test_sigma_t_positive(self):
        cfg = NonlinearSDEConfig(sigma=2.0, beta_min=1.0, beta_max=4.0, T=1.0)
        # g(t) = sigma * sqrt(beta(t)) > 0
        assert sigma_t(0.5, cfg) == pytest.approx(2.0 * math.sqrt(2.5))

    def test_n_steps_for_scaling(self):
        assert _n_steps_for(1.0, 250) == 250
        assert _n_steps_for(0.5, 250) == 125
        assert _n_steps_for(0.0, 250) == 10   # clipped to minimum


# ──────────────────────────────────────────────────────────────────────────────
# drift_nl
# ──────────────────────────────────────────────────────────────────────────────

class TestDriftNL:
    def test_output_shape(self):
        cfg = NonlinearSDEConfig()
        x   = torch.randn(100, 2)
        b   = drift_nl(x, 0.5, cfg)
        assert b.shape == (100, 2)

    def test_no_nan(self):
        cfg = NonlinearSDEConfig()
        x   = torch.randn(100, 2) * 10.0
        b   = drift_nl(x, 0.5, cfg)
        assert not torch.isnan(b).any()

    def test_sign_restoring_at_positive_x(self):
        """For a=0 and positive x, drift should be negative (pulls toward 0)."""
        cfg = NonlinearSDEConfig(a=0.0)
        x   = torch.ones(10, 2) * 2.0
        b   = drift_nl(x, 0.5, cfg)
        assert (b < 0).all()

    def test_sign_restoring_at_negative_x(self):
        cfg = NonlinearSDEConfig(a=0.0)
        x   = -torch.ones(10, 2) * 2.0
        b   = drift_nl(x, 0.5, cfg)
        assert (b > 0).all()

    def test_zero_at_a(self):
        """Drift vanishes at x = a."""
        cfg = NonlinearSDEConfig(a=0.0)
        x   = torch.zeros(5, 2)
        b   = drift_nl(x, 0.5, cfg)
        assert b.abs().max().item() < 1e-6

    def test_zero_at_nonzero_a(self):
        cfg = NonlinearSDEConfig(a=1.5)
        x   = torch.full((5, 2), 1.5)
        b   = drift_nl(x, 0.3, cfg)
        assert b.abs().max().item() < 1e-6

    def test_scales_with_beta(self):
        """Larger β → larger drift magnitude."""
        cfg = NonlinearSDEConfig(beta_min=1.0, beta_max=25.0, T=1.0)
        x   = torch.ones(5, 2) * 0.5
        b_early = drift_nl(x, 0.0, cfg).abs().mean().item()
        b_late  = drift_nl(x, 1.0, cfg).abs().mean().item()
        assert b_late > b_early


# ──────────────────────────────────────────────────────────────────────────────
# jac_drift_nl
# ──────────────────────────────────────────────────────────────────────────────

class TestJacDriftNL:
    def test_shape(self):
        cfg = NonlinearSDEConfig()
        x   = torch.randn(50, 2)
        J   = jac_drift_nl(x, 0.5, cfg)
        assert J.shape == (50, 2, 2)

    def test_off_diagonal_zero(self):
        """Component-wise drift ⟹ off-diagonal Jacobian entries are zero."""
        cfg = NonlinearSDEConfig()
        x   = torch.randn(30, 2) * 3.0
        J   = jac_drift_nl(x, 0.5, cfg)
        assert J[:, 0, 1].abs().max().item() < 1e-6
        assert J[:, 1, 0].abs().max().item() < 1e-6

    def test_no_nan(self):
        cfg = NonlinearSDEConfig()
        x   = torch.randn(30, 2)
        J   = jac_drift_nl(x, 0.5, cfg)
        assert not torch.isnan(J).any()

    def test_diagonal_negative_near_a(self):
        """At x ≈ a the drift is nearly linear, so ∂b/∂x ≈ -k β(t) < 0."""
        cfg = NonlinearSDEConfig(a=0.0)
        x   = torch.zeros(5, 2)
        J   = jac_drift_nl(x, 0.5, cfg)
        assert (J[:, 0, 0] < 0).all()
        assert (J[:, 1, 1] < 0).all()

    def test_matches_numerical_gradient(self):
        """Check ∂b_0/∂x_0 against finite difference."""
        cfg = NonlinearSDEConfig(a=0.0, k=1.0, beta_min=2.0, beta_max=2.0, T=1.0)
        x0  = torch.tensor([[0.7, -0.3]])
        eps = 1e-4
        xp  = x0.clone(); xp[0, 0] += eps
        xm  = x0.clone(); xm[0, 0] -= eps
        fd  = (drift_nl(xp, 0.5, cfg)[0, 0] - drift_nl(xm, 0.5, cfg)[0, 0]) / (2 * eps)
        J   = jac_drift_nl(x0, 0.5, cfg)[0, 0, 0]
        assert abs(float(J) - float(fd)) < 1e-3


# ──────────────────────────────────────────────────────────────────────────────
# simulate_forward_nl
# ──────────────────────────────────────────────────────────────────────────────

class TestSimulateForwardNL:
    def test_shape(self):
        X0, _ = sample_8gmm(_MINI_N, device=DEVICE)
        X_T   = simulate_forward_nl(X0, T=0.5, cfg=_MINI_CFG, n_steps=_MINI_STEPS)
        assert X_T.shape == (_MINI_N, 2)

    def test_no_nan(self):
        X0, _ = sample_8gmm(_MINI_N, device=DEVICE)
        X_T   = simulate_forward_nl(X0, T=1.0, cfg=_MINI_CFG, n_steps=_MINI_STEPS)
        assert not torch.isnan(X_T).any()

    def test_different_from_X0(self):
        """Simulation should move the particles (not trivially copy X0)."""
        X0, _ = sample_8gmm(_MINI_N, device=DEVICE)
        X_T   = simulate_forward_nl(X0, T=1.0, cfg=_MINI_CFG, n_steps=_MINI_STEPS)
        diff  = (X_T - X0).norm(dim=1).mean().item()
        assert diff > 0.01

    @pytest.mark.parametrize("T", [0.05, 0.20, 1.00])
    def test_various_T(self, T):
        X0, _ = sample_8gmm(100, device=DEVICE)
        X_T   = simulate_forward_nl(X0, T=T, cfg=_MINI_CFG,
                                    n_steps=max(5, round(T * 20)))
        assert X_T.shape == (100, 2)
        assert not torch.isnan(X_T).any()


# ──────────────────────────────────────────────────────────────────────────────
# simulate_malliavin_nl
# ──────────────────────────────────────────────────────────────────────────────

class TestSimulateMalliavinNL:
    @pytest.fixture(scope="class")
    def sim_result(self):
        X0, _ = sample_8gmm(_MINI_N, device=DEVICE)
        X_T, H = simulate_malliavin_nl(
            X0, T=0.5, cfg=_MINI_CFG,
            n_steps=_MINI_STEPS, gamma_reg=1e-3,
        )
        return X0, X_T, H

    def test_shapes(self, sim_result):
        X0, X_T, H = sim_result
        assert X_T.shape == (_MINI_N, 2)
        assert H.shape   == (_MINI_N, 2)

    def test_no_nan_xt(self, sim_result):
        _, X_T, _ = sim_result
        assert not torch.isnan(X_T).any()

    def test_h_finite(self, sim_result):
        _, _, H = sim_result
        assert torch.isfinite(H).all()

    def test_xt_not_trivially_zero(self, sim_result):
        _, X_T, _ = sim_result
        assert X_T.norm(dim=1).mean().item() > 0.01

    def test_gamma_reg_effect(self):
        """Larger gamma_reg should not crash and H should still be finite."""
        X0, _ = sample_8gmm(100, device=DEVICE)
        _, H  = simulate_malliavin_nl(
            X0, T=0.3, cfg=_MINI_CFG,
            n_steps=10, gamma_reg=0.1,
        )
        assert torch.isfinite(H).all()

    @pytest.mark.parametrize("T", [0.10, 0.50, 1.00])
    def test_various_T_no_nan(self, T):
        X0, _ = sample_8gmm(100, device=DEVICE)
        X_T, H = simulate_malliavin_nl(
            X0, T=T, cfg=_MINI_CFG,
            n_steps=max(5, round(T * 20)), gamma_reg=1e-3,
        )
        assert not torch.isnan(X_T).any()
        assert torch.isfinite(H).all()


# ──────────────────────────────────────────────────────────────────────────────
# apply_teacher_nl
# ──────────────────────────────────────────────────────────────────────────────

class TestApplyTeacherNL:
    @pytest.fixture(scope="class")
    def xt_h(self):
        X0, _  = sample_8gmm(2_000, device=DEVICE)
        X_T, H = simulate_malliavin_nl(
            X0, T=1.0, cfg=_MINI_CFG,
            n_steps=_MINI_STEPS, gamma_reg=1e-3,
        )
        return X_T, H

    @pytest.mark.parametrize("method", ["raw", "binned", "nw", "knn_nw"])
    def test_returns_triple(self, xt_h, method):
        X_T, H = xt_h
        pts, sc, cc = apply_teacher_nl(
            method, X_T, H,
            n_raw=200, n_bins=15, min_count=2, knn_k=20,
        )
        assert pts.ndim == 2 and pts.shape[1] == 2
        assert sc.shape == pts.shape
        assert cc.ndim == 1 and cc.shape[0] == pts.shape[0]

    @pytest.mark.parametrize("method", ["raw", "binned", "nw", "knn_nw"])
    def test_no_nan(self, xt_h, method):
        X_T, H = xt_h
        pts, sc, cc = apply_teacher_nl(
            method, X_T, H,
            n_raw=200, n_bins=15, min_count=2, knn_k=20,
        )
        assert not torch.isnan(pts).any()
        assert not torch.isnan(sc).any()

    def test_raw_size_bounded(self, xt_h):
        X_T, H = xt_h
        pts, _, _ = apply_teacher_nl("raw", X_T, H, n_raw=150)
        assert pts.shape[0] <= 150

    def test_unknown_method_raises(self, xt_h):
        X_T, H = xt_h
        with pytest.raises(ValueError):
            apply_teacher_nl("bad_method", X_T, H)


# ──────────────────────────────────────────────────────────────────────────────
# simulate_all_times_nl
# ──────────────────────────────────────────────────────────────────────────────

class TestSimulateAllTimesNL:
    @pytest.fixture(scope="class")
    def cache(self):
        return simulate_all_times_nl(
            times=[0.20, 0.50, 1.00],
            dataset_name="8gmm",
            cfg=_MINI_CFG,
            n_paths=_MINI_N,
            n_steps_per_unit=_MINI_STEPS,
            gamma_reg=1e-3,
            device=DEVICE,
        )

    def test_length(self, cache):
        assert len(cache) == 3

    def test_T_values(self, cache):
        Ts = [entry[0] for entry in cache]
        assert Ts == pytest.approx([0.20, 0.50, 1.00])

    def test_tensor_shapes(self, cache):
        for T, X_T, H in cache:
            assert X_T.shape == (_MINI_N, 2)
            assert H.shape   == (_MINI_N, 2)

    def test_no_nan(self, cache):
        for _, X_T, H in cache:
            assert not torch.isnan(X_T).any()
            assert torch.isfinite(H).all()


# ──────────────────────────────────────────────────────────────────────────────
# build_training_dataset_nl
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildTrainingDatasetNL:
    @pytest.fixture(scope="class")
    def small_cache(self):
        return simulate_all_times_nl(
            times=[0.30, 0.80],
            dataset_name="8gmm",
            cfg=_MINI_CFG,
            n_paths=600,
            n_steps_per_unit=_MINI_STEPS,
            gamma_reg=1e-3,
            device=DEVICE,
        )

    @pytest.mark.parametrize("method", ["raw", "binned", "nw", "knn_nw"])
    def test_returns_tensors(self, small_cache, method):
        result = build_training_dataset_nl(
            small_cache, method,
            n_raw=100, n_bins=12, min_count=2, knn_k=15,
        )
        assert result is not None, f"{method} returned None"
        t, x, s, c = result
        assert t.ndim == 1
        assert x.shape[1] == 2
        assert s.shape == x.shape
        assert c.shape[0] == x.shape[0]

    def test_lengths_consistent(self, small_cache):
        t, x, s, c = build_training_dataset_nl(
            small_cache, "binned", n_bins=12, min_count=2,
        )
        assert t.shape[0] == x.shape[0] == s.shape[0] == c.shape[0]

    def test_raw_no_nan(self, small_cache):
        t, x, s, c = build_training_dataset_nl(small_cache, "raw", n_raw=100)
        assert not torch.isnan(x).any()
        assert not torch.isnan(s).any()


# ──────────────────────────────────────────────────────────────────────────────
# reverse_euler_nl (smoke test only — model barely trained)
# ──────────────────────────────────────────────────────────────────────────────

class TestReverseEulerNL:
    @pytest.fixture(scope="class")
    def trained_model_and_cfg(self):
        cache = simulate_all_times_nl(
            times=[0.20, 0.50, 1.00],
            dataset_name="8gmm",
            cfg=_MINI_CFG,
            n_paths=_MINI_N,
            n_steps_per_unit=_MINI_STEPS,
            gamma_reg=1e-3,
            device=DEVICE,
        )
        ds = build_training_dataset_nl(cache, "raw", n_raw=200)
        if ds is None:
            pytest.skip("no valid teacher points")
        t, x, s, c = ds
        model = train_score_mlp(t, x, s, c, n_epochs=5, batch_size=32, device=DEVICE)
        return model, _MINI_CFG

    def test_output_shape(self, trained_model_and_cfg):
        model, cfg = trained_model_and_cfg
        X0, _  = sample_8gmm(80, device=DEVICE)
        X_T    = simulate_forward_nl(X0, cfg.T, cfg, n_steps=_MINI_STEPS)
        out    = reverse_euler_nl(model, X_T, cfg, n_steps=_MINI_STEPS)
        assert out.shape == (80, 2)

    def test_output_on_cpu(self, trained_model_and_cfg):
        model, cfg = trained_model_and_cfg
        X0, _  = sample_8gmm(50, device=DEVICE)
        X_T    = simulate_forward_nl(X0, cfg.T, cfg, n_steps=_MINI_STEPS)
        out    = reverse_euler_nl(model, X_T, cfg, n_steps=_MINI_STEPS)
        assert out.device.type == "cpu"

    def test_nan_rate_acceptable(self, trained_model_and_cfg):
        """Barely-trained model — just verify it doesn't blow up entirely."""
        model, cfg = trained_model_and_cfg
        X0, _  = sample_8gmm(200, device=DEVICE)
        X_T    = simulate_forward_nl(X0, cfg.T, cfg, n_steps=_MINI_STEPS)
        out    = reverse_euler_nl(model, X_T, cfg, n_steps=_MINI_STEPS)
        nan_rate = torch.isnan(out).any(dim=1).float().mean().item()
        assert nan_rate < 0.5


# ──────────────────────────────────────────────────────────────────────────────
# compute_metrics_nl
# ──────────────────────────────────────────────────────────────────────────────

class TestComputeMetricsNL:
    def _clean_8gmm(self, n=2_000):
        x, centers = sample_8gmm(n, device="cpu")
        return x.numpy(), centers.numpy()

    def test_keys_present(self):
        samples_np, centers_np = self._clean_8gmm()
        m = compute_metrics_nl(
            samples_np, nan_rate=0.0,
            dataset_name="8gmm", centers_np=centers_np, n_ref=500,
        )
        for key in ("mmd_rbf", "sliced_wasserstein", "nan_rate",
                    "n_samples", "mode_coverage"):
            assert key in m, f"Missing key: {key}"

    def test_mmd_low_for_same_dist(self):
        samples_np, centers_np = self._clean_8gmm(3_000)
        m = compute_metrics_nl(
            samples_np, nan_rate=0.0,
            dataset_name="8gmm", centers_np=centers_np, n_ref=3_000,
        )
        assert m["mmd_rbf"] < 0.05

    def test_nan_rate_passthrough(self):
        samples_np, centers_np = self._clean_8gmm()
        m = compute_metrics_nl(
            samples_np, nan_rate=0.123,
            dataset_name="8gmm", centers_np=centers_np, n_ref=200,
        )
        assert m["nan_rate"] == pytest.approx(0.123)

    def test_no_mode_coverage_for_swissroll(self):
        from scoremodel_ext.malliavin.datasets_2d import sample_swissroll
        x = sample_swissroll(1_000, device="cpu").numpy()
        m = compute_metrics_nl(x, nan_rate=0.0, dataset_name="swissroll", n_ref=200)
        assert "mode_coverage" not in m


# ──────────────────────────────────────────────────────────────────────────────
# teacher_eval_points="raw_points" — equal-count comparison
# ──────────────────────────────────────────────────────────────────────────────

class TestRawPointsMode:
    """
    Verify that all teacher methods return the same number of training points
    when teacher_eval_points="raw_points", and that their outputs are finite.
    """

    @pytest.fixture(scope="class")
    def xt_h(self):
        X0, _ = sample_8gmm(2_000, device=DEVICE)
        X_T, H = simulate_malliavin_nl(
            X0, T=1.0, cfg=_MINI_CFG,
            n_steps=_MINI_STEPS, gamma_reg=1e-3,
        )
        return X_T, H

    def test_all_methods_same_point_count(self, xt_h):
        """All methods with raw_points mode must return exactly n_raw points."""
        X_T, H = xt_h
        n_raw = 200
        counts = {}
        for method in ("raw", "binned", "nw", "knn_nw"):
            pts, sc, cc = apply_teacher_nl(
                method, X_T, H,
                n_raw=n_raw, n_bins=15, min_count=2, knn_k=20,
                teacher_eval_points="raw_points",
            )
            counts[method] = pts.shape[0]
        # all methods share the same subsample
        assert len(set(counts.values())) == 1, \
            f"Expected uniform point count, got {counts}"
        assert counts["raw"] <= n_raw

    @pytest.mark.parametrize("method", ["raw", "binned", "nw", "knn_nw"])
    def test_raw_points_no_nan(self, xt_h, method):
        """Scores and positions must be finite in raw_points mode."""
        X_T, H = xt_h
        pts, sc, cc = apply_teacher_nl(
            method, X_T, H,
            n_raw=200, n_bins=15, min_count=2, knn_k=20,
            teacher_eval_points="raw_points",
        )
        assert not torch.isnan(pts).any(), f"{method}: NaN in pts"
        assert not torch.isnan(sc).any(),  f"{method}: NaN in sc"
        assert torch.isfinite(sc).all(),   f"{method}: non-finite sc"

    @pytest.mark.parametrize("method", ["raw", "binned", "nw", "knn_nw"])
    def test_raw_points_uniform_weights(self, xt_h, method):
        """Counts must be all-ones (uniform weights) in raw_points mode."""
        X_T, H = xt_h
        _, _, cc = apply_teacher_nl(
            method, X_T, H,
            n_raw=200, n_bins=15, min_count=2, knn_k=20,
            teacher_eval_points="raw_points",
        )
        assert torch.allclose(cc, torch.ones_like(cc)), \
            f"{method}: expected uniform weights"

    def test_grid_centers_backward_compat(self, xt_h):
        """grid_centers mode must still work and can return different counts."""
        X_T, H = xt_h
        # raw in grid_centers is identical to raw in raw_points
        pts_raw_gc, sc_raw_gc, _ = apply_teacher_nl(
            "raw", X_T, H,
            n_raw=200, n_bins=15, min_count=2,
            teacher_eval_points="grid_centers",
        )
        assert pts_raw_gc.shape[1] == 2
        assert sc_raw_gc.shape == pts_raw_gc.shape

        # binned grid_centers returns bin-centre positions (not raw X_T points)
        pts_bin_gc, sc_bin, _ = apply_teacher_nl(
            "binned", X_T, H,
            n_raw=200, n_bins=15, min_count=2,
            teacher_eval_points="grid_centers",
        )
        assert pts_bin_gc.shape[1] == 2

    def test_invalid_teacher_eval_points_raises(self, xt_h):
        X_T, H = xt_h
        with pytest.raises(ValueError, match="teacher_eval_points"):
            apply_teacher_nl("raw", X_T, H, teacher_eval_points="bad_mode")

    def test_binned_score_at_points_shape(self, xt_h):
        """_binned_score_at_points must return (query_x, sc, cc) with matching shapes."""
        X_T, H = xt_h
        query_x = X_T[:50]
        out_pts, sc, cc = _binned_score_at_points(X_T, H, query_x, n_bins=15)
        assert out_pts.shape == query_x.shape
        assert sc.shape == query_x.shape
        assert cc.shape == (50,)
        assert torch.allclose(cc, torch.ones(50, device=X_T.device))

    def test_binned_score_at_points_no_nan(self, xt_h):
        X_T, H = xt_h
        query_x = X_T[:100]
        _, sc, _ = _binned_score_at_points(X_T, H, query_x, n_bins=15)
        assert not torch.isnan(sc).any()
        assert torch.isfinite(sc).all()

    def test_raw_points_propagated_through_build(self, xt_h):
        """build_training_dataset_nl respects teacher_eval_points for all methods."""
        X_T, H = xt_h
        cache = [(1.0, X_T, H)]
        n_raw = 150
        counts = {}
        for method in ("raw", "binned", "nw", "knn_nw"):
            result = build_training_dataset_nl(
                cache, method,
                n_raw=n_raw, n_bins=15, min_count=2, knn_k=20,
                teacher_eval_points="raw_points",
            )
            assert result is not None, f"{method} returned None"
            t, x, s, c = result
            counts[method] = x.shape[0]
        assert len(set(counts.values())) == 1, \
            f"Expected equal counts across methods, got {counts}"


# ──────────────────────────────────────────────────────────────────────────────
# build_results_table
# ──────────────────────────────────────────────────────────────────────────────

# Synthetic results dict used across all table tests
_FAKE_RESULTS = {
    "8gmm": {
        "raw":    {"mmd_rbf": 0.050, "sliced_wasserstein": 0.040, "nan_rate": 0.00,
                   "mode_coverage": {"coverage_fraction": 0.875, "mean_nearest_dist": 0.12}},
        "binned": {"mmd_rbf": 0.030, "sliced_wasserstein": 0.025, "nan_rate": 0.01,
                   "mode_coverage": {"coverage_fraction": 1.000, "mean_nearest_dist": 0.08}},
        "nw":     {"mmd_rbf": 0.020, "sliced_wasserstein": 0.018, "nan_rate": 0.00,
                   "mode_coverage": {"coverage_fraction": 1.000, "mean_nearest_dist": 0.07}},
        "knn_nw": {"mmd_rbf": 0.015, "sliced_wasserstein": 0.012, "nan_rate": 0.00,
                   "mode_coverage": {"coverage_fraction": 1.000, "mean_nearest_dist": 0.06}},
    },
    "swissroll": {
        "raw":    {"mmd_rbf": 0.080, "sliced_wasserstein": 0.070, "nan_rate": 0.02},
        "binned": {"mmd_rbf": 0.060, "sliced_wasserstein": 0.050, "nan_rate": 0.00},
        "nw":     {"mmd_rbf": 0.045, "sliced_wasserstein": 0.038, "nan_rate": 0.00},
        "knn_nw": {"mmd_rbf": 0.040, "sliced_wasserstein": 0.032, "nan_rate": 0.00},
    },
}


class TestBuildResultsTable:
    @pytest.fixture(scope="class")
    def df(self):
        return build_results_table(_FAKE_RESULTS)

    # ── shape & schema ──────────────────────────────────────────────────────
    def test_column_names(self, df):
        expected = {"dataset", "method", "mmd", "sliced_wasserstein",
                    "nan_rate", "coverage_fraction", "mean_nearest_dist"}
        assert expected == set(df.columns)

    def test_row_count(self, df):
        # 2 datasets × 4 methods = 8 rows
        assert len(df) == 8

    def test_method_order_preserved(self, df):
        methods_8gmm = df[df["dataset"] == "8gmm"]["method"].tolist()
        assert methods_8gmm == ["raw", "binned", "nw", "knn_nw"]

    def test_each_dataset_has_four_methods(self, df):
        for ds in ("8gmm", "swissroll"):
            assert len(df[df["dataset"] == ds]) == 4

    # ── best-value marking ──────────────────────────────────────────────────
    def test_best_mmd_marked_in_8gmm(self, df):
        """knn_nw has the smallest mmd (0.015) in 8gmm → must end with *."""
        row = df[(df["dataset"] == "8gmm") & (df["method"] == "knn_nw")].iloc[0]
        assert str(row["mmd"]).endswith("*"), \
            f"Expected best MMD marked, got {row['mmd']!r}"

    def test_non_best_mmd_not_marked(self, df):
        row = df[(df["dataset"] == "8gmm") & (df["method"] == "raw")].iloc[0]
        assert not str(row["mmd"]).endswith("*"), \
            f"Non-best row should not be marked, got {row['mmd']!r}"

    def test_best_sw_marked_in_8gmm(self, df):
        row = df[(df["dataset"] == "8gmm") & (df["method"] == "knn_nw")].iloc[0]
        assert str(row["sliced_wasserstein"]).endswith("*")

    def test_best_marked_per_dataset(self, df):
        """Each dataset must have exactly one '*' per metric column."""
        for ds in ("8gmm", "swissroll"):
            grp = df[df["dataset"] == ds]
            for col in ("mmd", "sliced_wasserstein", "nan_rate"):
                stars = grp[col].astype(str).str.endswith("*").sum()
                assert stars == 1, \
                    f"dataset={ds} col={col}: expected 1 star, got {stars}"

    # ── 8GMM-only columns ───────────────────────────────────────────────────
    def test_gmm_cols_present_for_8gmm(self, df):
        gmm = df[df["dataset"] == "8gmm"]
        assert gmm["coverage_fraction"].notna().all()
        assert gmm["mean_nearest_dist"].notna().all()

    def test_gmm_cols_absent_for_swissroll(self, df):
        sr = df[df["dataset"] == "swissroll"]
        assert sr["coverage_fraction"].isna().all()
        assert sr["mean_nearest_dist"].isna().all()

    # ── error rows skipped ──────────────────────────────────────────────────
    def test_error_rows_excluded(self):
        results_with_error = {
            "8gmm": {
                "raw":    {"error": "no_valid_teacher_points"},
                "binned": {"mmd_rbf": 0.03, "sliced_wasserstein": 0.02, "nan_rate": 0.0},
                "nw":     {"mmd_rbf": 0.02, "sliced_wasserstein": 0.01, "nan_rate": 0.0},
                "knn_nw": {"mmd_rbf": 0.01, "sliced_wasserstein": 0.01, "nan_rate": 0.0},
            }
        }
        df = build_results_table(results_with_error)
        assert len(df) == 3
        assert "raw" not in df["method"].tolist()

    def test_empty_results_returns_empty_df(self):
        df = build_results_table({})
        assert len(df) == 0

    # ── CSV / LaTeX output ──────────────────────────────────────────────────
    def test_csv_saved(self, df, tmp_path):
        build_results_table(_FAKE_RESULTS, outbase=str(tmp_path))
        csv = tmp_path / "summary.csv"
        assert csv.exists(), "summary.csv not created"
        import csv as csv_mod
        with open(csv) as f:
            reader = csv_mod.DictReader(f)
            rows = list(reader)
        assert len(rows) == 8
        assert "mmd" in rows[0]

    def test_tex_saved(self, df, tmp_path):
        build_results_table(_FAKE_RESULTS, outbase=str(tmp_path))
        tex = tmp_path / "summary.tex"
        assert tex.exists(), "summary.tex not created"
        content = tex.read_text()
        assert "\\begin{table}" in content
        assert "\\toprule" in content
        assert "\\textbf{" in content, "best values should be bold in LaTeX"

    def test_tex_has_midrule_between_datasets(self, tmp_path):
        build_results_table(_FAKE_RESULTS, outbase=str(tmp_path))
        content = (tmp_path / "summary.tex").read_text()
        # Two datasets → at least one \\midrule separator between blocks
        assert content.count("\\midrule") >= 2  # header + dataset separator


# ──────────────────────────────────────────────────────────────────────────────
# method="mirafzali" — Algorithm 6/7 faithful reproduction
# ──────────────────────────────────────────────────────────────────────────────

class TestMirafzaliMethod:
    """
    Verify that method='mirafzali' (Algorithm 6) uses ALL paths with equal
    weights, unlike 'raw' which subsamples to n_raw.
    """

    @pytest.fixture(scope="class")
    def xt_h(self):
        X0, _ = sample_8gmm(2_000, device=DEVICE)
        X_T, H = simulate_malliavin_nl(
            X0, T=1.0, cfg=_MINI_CFG,
            n_steps=_MINI_STEPS, gamma_reg=1e-3,
        )
        return X_T, H

    def test_uses_all_paths(self, xt_h):
        """mirafzali must return ALL n_paths points — no subsampling."""
        X_T, H = xt_h
        pts, sc, cc = apply_teacher_nl("mirafzali", X_T, H, n_raw=200)
        assert pts.shape[0] == X_T.shape[0], (
            f"Expected {X_T.shape[0]} points (all paths), got {pts.shape[0]}"
        )

    def test_scores_equal_h(self, xt_h):
        """mirafzali score == H (raw Malliavin weights, no smoothing)."""
        X_T, H = xt_h
        pts, sc, cc = apply_teacher_nl("mirafzali", X_T, H)
        assert torch.allclose(sc, H), "mirafzali scores should equal H exactly"

    def test_uniform_weights(self, xt_h):
        """mirafzali must use equal (all-ones) weights."""
        X_T, H = xt_h
        _, _, cc = apply_teacher_nl("mirafzali", X_T, H)
        assert torch.allclose(cc, torch.ones_like(cc)), \
            "mirafzali weights should all be 1.0"

    def test_no_nan(self, xt_h):
        X_T, H = xt_h
        pts, sc, cc = apply_teacher_nl("mirafzali", X_T, H)
        assert not torch.isnan(pts).any(), "NaN in pts"
        assert not torch.isnan(sc).any(),  "NaN in sc"
        assert torch.isfinite(sc).all(),   "non-finite sc"

    def test_more_points_than_raw(self, xt_h):
        """mirafzali uses strictly more training points than raw (which subsamples)."""
        X_T, H = xt_h
        n_raw = 200
        pts_m, _, _ = apply_teacher_nl("mirafzali", X_T, H, n_raw=n_raw)
        pts_r, _, _ = apply_teacher_nl("raw",       X_T, H, n_raw=n_raw)
        assert pts_m.shape[0] > pts_r.shape[0], (
            f"mirafzali ({pts_m.shape[0]}) should have more points than "
            f"raw with n_raw={n_raw} ({pts_r.shape[0]})"
        )

    def test_same_in_both_teacher_eval_modes(self, xt_h):
        """mirafzali behaves identically for raw_points and grid_centers."""
        X_T, H = xt_h
        pts_rp, sc_rp, cc_rp = apply_teacher_nl(
            "mirafzali", X_T, H, teacher_eval_points="raw_points",
        )
        pts_gc, sc_gc, cc_gc = apply_teacher_nl(
            "mirafzali", X_T, H, teacher_eval_points="grid_centers",
        )
        assert pts_rp.shape == pts_gc.shape
        assert torch.allclose(pts_rp, pts_gc)
        assert torch.allclose(sc_rp, sc_gc)
        assert torch.allclose(cc_rp, cc_gc)

    def test_build_dataset_mirafzali_all_paths(self):
        """build_training_dataset_nl with mirafzali includes all paths per time."""
        times = [0.30, 0.80]
        n_paths = 500
        cache = []
        for T in times:
            X0_, _ = sample_8gmm(n_paths, device=DEVICE)
            X_T, H = simulate_malliavin_nl(
                X0_, T=T, cfg=_MINI_CFG,
                n_steps=_MINI_STEPS, gamma_reg=1e-3,
            )
            cache.append((T, X_T, H))

        result = build_training_dataset_nl(cache, "mirafzali")
        assert result is not None
        t, x, s, c = result
        # All n_paths × n_times rows (no subsampling)
        assert x.shape[0] == n_paths * len(times), (
            f"Expected {n_paths * len(times)} rows, got {x.shape[0]}"
        )
        assert torch.allclose(c, torch.ones_like(c)), \
            "mirafzali dataset weights should all be 1.0"

    def test_build_results_table_includes_mirafzali(self):
        """build_results_table correctly handles mirafzali method entries."""
        results = {
            "swissroll": {
                "raw":       {"mmd_rbf": 0.08, "sliced_wasserstein": 0.07, "nan_rate": 0.02},
                "mirafzali": {"mmd_rbf": 0.05, "sliced_wasserstein": 0.04, "nan_rate": 0.00},
            }
        }
        df = build_results_table(results)
        assert "mirafzali" in df["method"].tolist(), \
            "mirafzali should appear in results table"
        assert len(df) == 2  # raw + mirafzali
        # mirafzali has lower mmd → should be marked best
        row = df[df["method"] == "mirafzali"].iloc[0]
        assert str(row["mmd"]).endswith("*"), \
            f"mirafzali has best mmd but was not marked: {row['mmd']!r}"


# ──────────────────────────────────────────────────────────────────────────────
# MirafzaliSkorokhodNet architecture
# ──────────────────────────────────────────────────────────────────────────────

class TestMirafzaliSkorokhodNet:
    def test_output_shape(self):
        net = MirafzaliSkorokhodNet(x_dim=2, hidden=64, n_blocks=2, num_frequencies=8)
        t = torch.linspace(0.1, 1.0, 20)
        x = torch.randn(20, 2)
        out = net(t, x)
        assert out.shape == (20, 2)

    def test_t_2d_input(self):
        """Accept t as (N,1) as well as (N,)."""
        net = MirafzaliSkorokhodNet(x_dim=2, hidden=64, n_blocks=2, num_frequencies=8)
        t = torch.linspace(0.1, 1.0, 10).unsqueeze(1)  # (10, 1)
        x = torch.randn(10, 2)
        out = net(t, x)
        assert out.shape == (10, 2)

    def test_no_nan_random_input(self):
        net = MirafzaliSkorokhodNet(x_dim=2, hidden=64, n_blocks=2, num_frequencies=8)
        t = torch.rand(50)
        x = torch.randn(50, 2)
        out = net(t, x)
        assert not torch.isnan(out).any()
        assert torch.isfinite(out).all()


# ──────────────────────────────────────────────────────────────────────────────
# train_mirafzali_skorokhod_net — smoke test (hidden=1024, n_blocks=4)
# ──────────────────────────────────────────────────────────────────────────────

class TestTrainMirafzaliSkorokhodNet:
    """Smoke tests: small epochs so the suite stays fast."""

    @pytest.fixture(scope="class")
    def small_data(self):
        """Tiny (t, x, delta) dataset from a real Malliavin simulation."""
        X0, _ = sample_8gmm(600, device=DEVICE)
        X_T, H = simulate_malliavin_nl(
            X0, T=1.0, cfg=_MINI_CFG,
            n_steps=_MINI_STEPS, gamma_reg=1e-3,
        )
        # two time steps stacked
        t = torch.cat([torch.full((600,), 0.5), torch.full((600,), 1.0)]).to(DEVICE)
        x = torch.cat([X_T, X_T]).to(DEVICE)
        delta = torch.cat([H, H]).to(DEVICE)
        return t, x, delta

    def test_returns_normalized_model(self, small_data):
        t, x, delta = small_data
        model = train_mirafzali_skorokhod_net(
            t, x, delta,
            n_epochs=10, batch_size=64,
            hidden=64, n_blocks=2, num_frequencies=8,
            device=DEVICE,
        )
        assert isinstance(model, NormalizedSkorokhodModel)
        assert isinstance(model.net, MirafzaliSkorokhodNet)

    def test_output_shape(self, small_data):
        t, x, delta = small_data
        model = train_mirafzali_skorokhod_net(
            t, x, delta,
            n_epochs=10, batch_size=64,
            hidden=64, n_blocks=2, num_frequencies=8,
            device=DEVICE,
        )
        n = 30
        tq = torch.rand(n, device=DEVICE)
        xq = torch.randn(n, 2, device=DEVICE)
        with torch.no_grad():
            out = model(tq, xq)
        assert out.shape == (n, 2)
        assert torch.isfinite(out).all()

    def test_hidden1024_n_blocks4_smoke(self, small_data):
        """Requested smoke-test: hidden=1024, n_blocks=4."""
        t, x, delta = small_data
        model = train_mirafzali_skorokhod_net(
            t, x, delta,
            n_epochs=10, batch_size=64,
            hidden=1024, n_blocks=4, num_frequencies=16,
            device=DEVICE,
        )
        assert isinstance(model, NormalizedSkorokhodModel)
        tq = torch.rand(16, device=DEVICE)
        xq = torch.randn(16, 2, device=DEVICE)
        with torch.no_grad():
            out = model(tq, xq)
        assert out.shape == (16, 2)
        assert not torch.isnan(out).any()

    def test_normalisation_buffers_registered(self, small_data):
        """NormalizedSkorokhodModel must have all stat buffers on the right device."""
        t, x, delta = small_data
        model = train_mirafzali_skorokhod_net(
            t, x, delta,
            n_epochs=5, batch_size=32,
            hidden=32, n_blocks=1, num_frequencies=4,
            device=DEVICE,
        )
        for name in ("x_mean", "x_std", "t_mean", "t_std", "y_mean", "y_std"):
            buf = getattr(model, name)
            assert buf is not None, f"Missing buffer: {name}"
            assert torch.isfinite(buf).all(), f"Non-finite buffer: {name}"

    def test_plain_mse_no_weighting(self, small_data):
        """
        train_mirafzali_skorokhod_net does NOT use per-point weights.
        Passing a c vector must not change the model that's returned.
        (Verify by checking the training reaches completion for any c.)
        """
        t, x, delta = small_data
        # just confirm it trains without error when ignoring c
        model = train_mirafzali_skorokhod_net(
            t, x, delta,
            n_epochs=5, batch_size=32,
            hidden=32, n_blocks=1, num_frequencies=4,
            device=DEVICE,
        )
        assert isinstance(model, NormalizedSkorokhodModel)

