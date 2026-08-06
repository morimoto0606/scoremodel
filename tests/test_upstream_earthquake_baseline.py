import json
from pathlib import Path

from scripts.postprocess_earthquake_baseline_matrix import _load_metrics
from scripts.postprocess_upstream_earthquake_baseline import (
    _last_train_loss,
    _sample_metadata,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = REPOSITORY_ROOT / "upstream" / "riemannian-score-sde"


def test_upstream_heat_and_varadhan_configs_select_existing_dsm_branches():
    heat = (UPSTREAM_ROOT / "config/experiment/earthquake_heat.yaml").read_text()
    varadhan = (
        UPSTREAM_ROOT / "config/experiment/earthquake_varadhan.yaml"
    ).read_text()
    loss_source = (UPSTREAM_ROOT / "riemannian_score_sde/losses.py").read_text()

    assert "teacher: heat" in heat
    assert "n_max: 5" in heat
    assert "teacher: varadhan" in varadhan
    assert "n_max: -1" in varadhan
    assert 'kwargs["n_max"] <= -1' in loss_source
    assert "sde.varhadan_exp" in loss_source
    assert "sde.grad_marginal_log_prob" in loss_source


def test_upstream_training_loss_and_sample_metadata_adapters(tmp_path):
    metrics_dir = tmp_path / "logs" / "version_0"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "metrics.csv").write_text(
        "train/loss,step\n1.25,0\n0.75,50\n",
        encoding="utf-8",
    )
    assert _last_train_loss(tmp_path) == 0.75

    samples_path = tmp_path / "generated_samples.npy"
    metadata = {
        "coordinate_system": "upstream-earthquake-antipodal",
        "reverse_steps": 100,
        "epsilon": 0.001,
    }
    samples_path.with_suffix(".json").write_text(json.dumps(metadata))
    assert _sample_metadata(samples_path) == metadata


def test_baseline_matrix_metrics_loader_preserves_method_metrics(tmp_path):
    path = tmp_path / "metrics.json"
    expected = {
        "teacher": "heat",
        "s2_rbf_mmd": 0.01,
        "nearest_neighbor_geodesic_mean": 0.1,
    }
    path.write_text(json.dumps(expected), encoding="utf-8")
    assert _load_metrics(path) == expected
