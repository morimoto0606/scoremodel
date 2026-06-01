"""
Tests for simulate_malliavin_nl dispatcher and both correction modes.
"""

import torch
import pytest

from scoremodel_ext.malliavin.sde_nonlinear import (
    NonlinearSDEConfig,
    simulate_malliavin_nl,
    simulate_malliavin_nl_approx,
    simulate_malliavin_nl_mirafzali_full,
)

SMALL_N = 32
SMALL_STEPS = 10


@pytest.fixture
def cfg():
    return NonlinearSDEConfig()


@pytest.fixture
def X0(cfg):
    torch.manual_seed(0)
    return torch.randn(SMALL_N, 2)


# ── approx mode ───────────────────────────────────────────────────────────────

def test_approx_output_shape(cfg, X0):
    X_T, H = simulate_malliavin_nl(X0, cfg.T, cfg, n_steps=SMALL_STEPS, correction="approx")
    assert X_T.shape == (SMALL_N, 2)
    assert H.shape == (SMALL_N, 2)


def test_approx_finite(cfg, X0):
    X_T, H = simulate_malliavin_nl(X0, cfg.T, cfg, n_steps=SMALL_STEPS, correction="approx")
    assert torch.isfinite(X_T).all(), "X_T has inf/nan (approx)"
    assert torch.isfinite(H).all(), "H has inf/nan (approx)"


def test_approx_no_nan(cfg, X0):
    X_T, H = simulate_malliavin_nl(X0, cfg.T, cfg, n_steps=SMALL_STEPS, correction="approx")
    assert not torch.isnan(H).any(), "H has NaN (approx)"


def test_approx_consistent_with_direct(cfg, X0):
    """Dispatcher and direct function must produce identical results for same seed."""
    torch.manual_seed(42)
    X_T1, H1 = simulate_malliavin_nl_approx(X0, cfg.T, cfg, n_steps=SMALL_STEPS)
    torch.manual_seed(42)
    X_T2, H2 = simulate_malliavin_nl(X0, cfg.T, cfg, n_steps=SMALL_STEPS, correction="approx")
    assert torch.allclose(X_T1, X_T2)
    assert torch.allclose(H1, H2)


# ── mirafzali_full mode ───────────────────────────────────────────────────────

def test_full_output_shape(cfg, X0):
    X_T, H = simulate_malliavin_nl(X0, cfg.T, cfg, n_steps=SMALL_STEPS, correction="mirafzali_full")
    assert X_T.shape == (SMALL_N, 2)
    assert H.shape == (SMALL_N, 2)


def test_full_finite(cfg, X0):
    X_T, H = simulate_malliavin_nl(X0, cfg.T, cfg, n_steps=SMALL_STEPS, correction="mirafzali_full")
    assert torch.isfinite(X_T).all(), "X_T has inf (full)"
    assert torch.isfinite(H).all(), "H has inf (full)"


def test_full_no_nan(cfg, X0):
    X_T, H = simulate_malliavin_nl(X0, cfg.T, cfg, n_steps=SMALL_STEPS, correction="mirafzali_full")
    assert not torch.isnan(H).any(), "H has NaN (full)"


def test_full_consistent_with_direct(cfg, X0):
    """Dispatcher and direct function must produce identical results for same seed."""
    torch.manual_seed(7)
    X_T1, H1 = simulate_malliavin_nl_mirafzali_full(X0, cfg.T, cfg, n_steps=SMALL_STEPS)
    torch.manual_seed(7)
    X_T2, H2 = simulate_malliavin_nl(X0, cfg.T, cfg, n_steps=SMALL_STEPS, correction="mirafzali_full")
    assert torch.allclose(X_T1, X_T2)
    assert torch.allclose(H1, H2)


# ── dispatcher error handling ─────────────────────────────────────────────────

def test_invalid_correction_raises(cfg, X0):
    with pytest.raises(ValueError, match="Unknown correction"):
        simulate_malliavin_nl(X0, cfg.T, cfg, n_steps=SMALL_STEPS, correction="bad_value")
