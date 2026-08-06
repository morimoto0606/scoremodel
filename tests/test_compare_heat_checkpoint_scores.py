import csv
import json
import math

import pytest
import torch

from scripts.compare_heat_checkpoint_scores import (
    compute_score_metrics,
    save_score_comparison,
)


def test_compute_score_metrics_cosine_relative_error_and_norm_ratio():
    upstream = torch.tensor(
        [[1.0, 0.0], [0.0, 2.0], [1.0, 0.0]], dtype=torch.float64
    )
    ext = torch.tensor(
        [[1.0, 0.0], [0.0, 4.0], [-1.0, 0.0]], dtype=torch.float64
    )

    metrics = compute_score_metrics(upstream, ext, epsilon=1e-15)

    assert metrics["cosine_mean"] == pytest.approx(1.0 / 3.0)
    assert metrics["cosine_std"] == pytest.approx(math.sqrt(8.0 / 9.0))
    assert metrics["cosine_min"] == pytest.approx(-1.0)
    assert metrics["relative_error_mean"] == pytest.approx(1.0)
    assert metrics["relative_error_std"] == pytest.approx(math.sqrt(2.0 / 3.0))
    assert metrics["norm_ratio_mean"] == pytest.approx(4.0 / 3.0)
    assert metrics["norm_ratio_std"] == pytest.approx(math.sqrt(2.0 / 9.0))


def test_save_score_comparison_writes_json_and_csv(tmp_path):
    results = [
        {
            "t": 0.1,
            "cosine_mean": 0.9,
            "cosine_std": 0.1,
            "cosine_min": 0.5,
            "relative_error_mean": 0.2,
            "relative_error_std": 0.03,
            "norm_ratio_mean": 1.1,
            "norm_ratio_std": 0.04,
        }
    ]

    json_path, csv_path = save_score_comparison(
        tmp_path,
        upstream_checkpoint="upstream/ckpt",
        ext_checkpoint="ext/model.pt",
        sample_count=4096,
        results=results,
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload == {
        "upstream_checkpoint": "upstream/ckpt",
        "ext_checkpoint": "ext/model.pt",
        "sample_count": 4096,
        "results": results,
    }
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert float(rows[0]["t"]) == 0.1
    assert float(rows[0]["cosine_mean"]) == 0.9

