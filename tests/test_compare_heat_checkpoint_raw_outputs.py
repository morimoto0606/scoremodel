import csv
import json

import pytest
import torch

from scripts.compare_heat_checkpoint_raw_outputs import (
    compute_raw_output_metrics,
    evaluate_raw_outputs,
    save_raw_output_comparison,
)


def test_compute_raw_output_metrics():
    upstream = torch.tensor([[3.0, 0.0], [0.0, 4.0]], dtype=torch.float64)
    ext_scaled = 2.0 * upstream

    metrics = compute_raw_output_metrics(upstream, ext_scaled)

    assert metrics["cosine_similarity"] == pytest.approx(1.0)
    assert metrics["cosine_similarity_std"] == pytest.approx(0.0)
    assert metrics["relative_l2_error"] == pytest.approx(1.0)
    assert metrics["norm_ratio"] == pytest.approx(2.0)
    assert metrics["upstream_raw_output_l2_norm"] == pytest.approx(5.0)
    assert metrics["scaled_ext_score_l2_norm"] == pytest.approx(10.0)


def test_evaluate_raw_outputs_multiplies_ext_score_by_sigma_once():
    points = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)

    def upstream_raw(times, locations):
        sigma = torch.sqrt(1.0 - torch.exp(-(0.001 * times + 2.4995 * times**2)))
        return sigma[:, None] * locations

    def ext_effective(times, locations):
        return locations

    result = evaluate_raw_outputs(
        points,
        [0.1],
        upstream_raw,
        ext_effective,
    )[0]

    assert result["cosine_similarity"] == pytest.approx(1.0)
    assert result["relative_l2_error"] == pytest.approx(0.0, abs=1e-12)
    assert result["norm_ratio"] == pytest.approx(1.0)


def test_save_raw_output_comparison(tmp_path):
    results = [
        {
            "t": 0.1,
            "sigma": 0.2,
            "cosine_similarity": 0.9,
            "cosine_similarity_std": 0.1,
            "cosine_similarity_min": 0.5,
            "relative_l2_error": 0.3,
            "norm_ratio": 1.1,
            "upstream_raw_output_l2_norm": 2.0,
            "scaled_ext_score_l2_norm": 2.2,
        }
    ]

    json_path, csv_path = save_raw_output_comparison(
        tmp_path,
        upstream_checkpoint="upstream/ckpt",
        ext_checkpoint="ext/model.pt",
        sample_count=4096,
        results=results,
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["sample_count"] == 4096
    assert payload["comparison"] == "N_up(x,t) vs sigma(t) * ext_effective_score(x,t)"
    assert payload["results"] == results
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert float(rows[0]["relative_l2_error"]) == pytest.approx(0.3)
