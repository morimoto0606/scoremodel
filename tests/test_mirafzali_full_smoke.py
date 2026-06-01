import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scoremodel_ext.malliavin.datasets_2d import sample_8gmm
from scoremodel_ext.malliavin.experiment_mirafzali_nonlinear import run_mirafzali_full_smoke
from scoremodel_ext.malliavin.sde_nonlinear import NonlinearSDEConfig, simulate_malliavin_nl


def test_run_mirafzali_full_smoke_pipeline(tmp_path):
    outbase = tmp_path / "mirafzali_full_smoke"
    metrics = run_mirafzali_full_smoke(
        dataset="swissroll",
        n_paths=128,
        n_epochs=1,
        batch_size=32,
        n_steps_per_unit=10,
        hidden=64,
        n_blocks=1,
        outbase=str(outbase),
        device="cpu",
    )

    assert isinstance(metrics, dict)
    assert "nan_rate" in metrics
    assert "mmd_rbf" in metrics
    assert "sliced_wasserstein" in metrics
    assert metrics["nan_rate"] <= 1.0

    metrics_path = outbase / "swissroll" / "mirafzali" / "metrics.json"
    assert metrics_path.exists()


def test_simulate_malliavin_nl_correction_modes_small():
    cfg = NonlinearSDEConfig(
        k=1.0,
        sigma=1.0,
        a=0.0,
        beta_min=1.0,
        beta_max=10.0,
        T=1.0,
    )
    n_paths = 32
    X0, _ = sample_8gmm(n_paths, device="cpu")

    torch.manual_seed(123)
    X_approx, H_approx = simulate_malliavin_nl(
        X0,
        T=0.2,
        cfg=cfg,
        n_steps=5,
        gamma_reg=1e-3,
        correction="approx",
    )

    torch.manual_seed(123)
    X_acorr, H_acorr = simulate_malliavin_nl(
        X0,
        T=0.2,
        cfg=cfg,
        n_steps=5,
        gamma_reg=1e-3,
        correction="a_correction",
    )

    torch.manual_seed(123)
    X_full, H_full = simulate_malliavin_nl(
        X0,
        T=0.2,
        cfg=cfg,
        n_steps=5,
        gamma_reg=1e-3,
        correction="mirafzali_full",
    )

    for Xt, Ht in ((X_approx, H_approx), (X_acorr, H_acorr), (X_full, H_full)):
        assert Xt.shape == (n_paths, 2)
        assert Ht.shape == (n_paths, 2)
        assert torch.isfinite(Ht).all()

    assert torch.norm(H_full - H_approx).item() > 1e-6
