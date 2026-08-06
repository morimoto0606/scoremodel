import json

import pytest
import torch
from torch import nn

from scripts.diagnose_heat_score_definition import (
    diagnose_score_definitions,
    field_diagnostics,
    pairwise_diagnostics,
    project_s2_tangent,
    save_diagnostic,
)


class _IdentityNet(nn.Module):
    def forward(self, times, points):
        del times
        return points


class _FakeNormalizedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = _IdentityNet()
        self.register_buffer("x_mean", torch.zeros(1, 3, dtype=torch.float64))
        self.register_buffer("x_std", torch.ones(1, 3, dtype=torch.float64))
        self.register_buffer("t_mean", torch.zeros(1, 1, dtype=torch.float64))
        self.register_buffer("t_std", torch.ones(1, 1, dtype=torch.float64))
        self.register_buffer("y_mean", torch.zeros(1, 3, dtype=torch.float64))
        self.register_buffer("y_std", torch.ones(1, 3, dtype=torch.float64))

    def forward(self, times, points):
        return self.net(times, points)


def test_tangent_projection_and_field_diagnostics():
    points = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    field = torch.tensor([[2.0, 3.0, 0.0]], dtype=torch.float64)

    projected = project_s2_tangent(points, field)
    diagnostics = field_diagnostics(points, projected)

    assert torch.equal(projected, torch.tensor([[0.0, 3.0, 0.0]], dtype=torch.float64))
    assert diagnostics["x_dot_score"]["max"] == pytest.approx(0.0)


def test_pairwise_diagnostics_reports_scale():
    reference = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float64)
    diagnostics = pairwise_diagnostics(reference, 3.0 * reference)

    assert diagnostics["cosine"]["mean"] == pytest.approx(1.0)
    assert diagnostics["norm_ratio"] == pytest.approx(3.0)
    assert diagnostics["least_squares_coordinate_scaling_factor"] == pytest.approx(3.0)
    assert diagnostics["residual_after_scalar_fit"] == pytest.approx(0.0, abs=1e-12)


def test_diagnosis_contains_both_coordinate_policies_and_sigma_hypotheses():
    points = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64
    )
    model = _FakeNormalizedModel()

    def upstream_raw(times, locations):
        tau = 0.001 * times + 2.4995 * times**2
        sigma = torch.sqrt(1.0 - torch.exp(-tau))
        return sigma[:, None] * locations

    row = diagnose_score_definitions(points, [0.1], upstream_raw, model)[0]

    assert set(row["comparisons"]) == {"same_numeric_xyz", "coordinate_aligned"}
    same = row["comparisons"]["same_numeric_xyz"]
    assert same["raw_output"]["N_up_vs_sigma_times_ext_wrapper"][
        "relative_l2_error"
    ] == pytest.approx(0.0, abs=1e-12)
    assert same["effective_score"]["N_up_over_sigma_vs_ext_wrapper"][
        "relative_l2_error"
    ] == pytest.approx(0.0, abs=1e-12)


def test_save_diagnostic_json(tmp_path):
    # save_diagnostic derives evidence, so use a minimal valid real diagnostic.
    points = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    model = _FakeNormalizedModel()

    def upstream_raw(times, locations):
        tau = 0.001 * times + 2.4995 * times**2
        sigma = torch.sqrt(1.0 - torch.exp(-tau))
        return sigma[:, None] * locations

    result = diagnose_score_definitions(points, [0.1], upstream_raw, model)
    output_path = save_diagnostic(
        tmp_path,
        upstream_checkpoint="upstream/ckpt",
        ext_checkpoint="ext/model.pt",
        data_path="quakes.csv",
        sample_count=1,
        sample_seed=0,
        results=result,
        normalization_buffers={"x_std": [[1.0, 1.0, 1.0]]},
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["reverse_sampler_used"] is False
    assert payload["coordinate_definition"]["coordinate_sign"] == -1.0
    assert payload["parameterization_evidence"]["same_numeric_xyz"][
        "lower_error_hypothesis"
    ] == "ext_output_is_effective_score"
