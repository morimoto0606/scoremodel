"""
pytest smoke tests for experiment_teacher_compare_1d (Spec 01).

Run with:
    cd /export/home/ymorimoto/github/scoremodel
    pytest tests/test_teacher_compare_1d.py -v
"""

import math
import numpy as np
import pytest
import torch

# allow running from project root or from tests/
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scoremodel_ext.malliavin.experiment_teacher_compare_1d import (
    simulate_malliavin_raw,
    binned_teacher_1d,
    nw_teacher_1d,
    knn_nw_teacher_1d,
    compare_teachers_at_T,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── unit tests ────────────────────────────────────────────────────────────────

class TestSimulateMalliavinRaw:
    def test_shapes(self):
        n = 1_000
        X_T, H = simulate_malliavin_raw(T=0.2, n_paths=n, n_steps=20, device=DEVICE)
        assert X_T.shape == (n,)
        assert H.shape   == (n,)

    def test_finite(self):
        X_T, H = simulate_malliavin_raw(T=0.2, n_paths=1_000, n_steps=20, device=DEVICE)
        assert torch.isfinite(X_T).all(), "X_T contains non-finite values"
        assert torch.isfinite(H).all(),   "H contains non-finite values"

    def test_x0_mean_crude(self):
        """Very short T: mean of X_T should be close to x0=1.5."""
        X_T, _ = simulate_malliavin_raw(T=0.02, n_paths=5_000, n_steps=10, device=DEVICE)
        assert abs(X_T.mean().item() - 1.5) < 0.3

    def test_longer_T_moves_mean(self):
        """Longer T: mean should be pulled toward 0 by drift."""
        X_T, _ = simulate_malliavin_raw(T=1.0, n_paths=5_000, n_steps=100, device=DEVICE)
        assert X_T.mean().item() < 1.0  # drift -x - x^3 pulls toward 0


class TestBinnedTeacher1d:
    def setup_method(self):
        torch.manual_seed(42)
        self.X_T, self.H = simulate_malliavin_raw(
            T=0.4, n_paths=20_000, n_steps=80, device=DEVICE
        )

    def test_shapes(self):
        centers, score, counts = binned_teacher_1d(self.X_T, self.H, n_bins=60, min_count=10)
        assert centers.shape == score.shape == counts.shape
        assert centers.ndim == 1
        assert centers.shape[0] > 0

    def test_centers_sorted(self):
        centers, _, _ = binned_teacher_1d(self.X_T, self.H, n_bins=60, min_count=10)
        diffs = centers[1:] - centers[:-1]
        assert (diffs > 0).all()

    def test_counts_above_min(self):
        min_count = 10
        _, _, counts = binned_teacher_1d(self.X_T, self.H, n_bins=60, min_count=min_count)
        assert (counts >= min_count).all()

    def test_finite_scores(self):
        _, score, _ = binned_teacher_1d(self.X_T, self.H, n_bins=60, min_count=10)
        assert torch.isfinite(score).all()


class TestNWTeacher1d:
    def setup_method(self):
        torch.manual_seed(0)
        self.X_T, self.H = simulate_malliavin_raw(
            T=0.4, n_paths=10_000, n_steps=60, device=DEVICE
        )
        self.centers, _, _ = binned_teacher_1d(self.X_T, self.H, n_bins=40, min_count=5)

    def test_shape(self):
        score = nw_teacher_1d(self.X_T, self.H, self.centers)
        assert score.shape == self.centers.shape

    def test_finite(self):
        score = nw_teacher_1d(self.X_T, self.H, self.centers)
        assert torch.isfinite(score).all()

    def test_explicit_bandwidth(self):
        score = nw_teacher_1d(self.X_T, self.H, self.centers, bandwidth=0.1)
        assert torch.isfinite(score).all()

    def test_not_all_zero(self):
        score = nw_teacher_1d(self.X_T, self.H, self.centers)
        assert score.abs().max() > 0.01


class TestKnnNWTeacher1d:
    def setup_method(self):
        torch.manual_seed(1)
        self.X_T, self.H = simulate_malliavin_raw(
            T=0.4, n_paths=10_000, n_steps=60, device=DEVICE
        )
        self.centers, _, _ = binned_teacher_1d(self.X_T, self.H, n_bins=40, min_count=5)

    def test_shape(self):
        score = knn_nw_teacher_1d(self.X_T, self.H, self.centers, k=100)
        assert score.shape == self.centers.shape

    def test_finite(self):
        score = knn_nw_teacher_1d(self.X_T, self.H, self.centers, k=100)
        assert torch.isfinite(score).all()

    def test_not_all_zero(self):
        score = knn_nw_teacher_1d(self.X_T, self.H, self.centers, k=100)
        assert score.abs().max() > 0.01


# ── integration smoke test ────────────────────────────────────────────────────

class TestCompareTeachersAtT:
    """Lightweight end-to-end smoke test (small n_paths / n_steps)."""

    def test_returns_finite_rmse(self):
        r = compare_teachers_at_T(
            T=0.4,
            n_paths=5_000,
            n_steps=40,
            n_bins=40,
            knn_k=100,
            device=DEVICE,
        )
        assert math.isfinite(r["rmse_bin"]),  "rmse_bin is not finite"
        assert math.isfinite(r["rmse_nw"]),   "rmse_nw is not finite"
        assert math.isfinite(r["rmse_knn"]),  "rmse_knn is not finite"

    def test_result_keys(self):
        r = compare_teachers_at_T(
            T=0.2, n_paths=3_000, n_steps=30, n_bins=30, knn_k=60, device=DEVICE
        )
        for key in ("T", "query_x", "valid", "score_pde",
                    "score_bin", "score_nw", "score_knn",
                    "rmse_bin", "rmse_nw", "rmse_knn"):
            assert key in r, f"missing key: {key}"

    def test_rmse_positive(self):
        r = compare_teachers_at_T(
            T=0.4, n_paths=5_000, n_steps=40, n_bins=40, knn_k=100, device=DEVICE
        )
        assert r["rmse_bin"] >= 0
        assert r["rmse_nw"]  >= 0
        assert r["rmse_knn"] >= 0
