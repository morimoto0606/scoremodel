import json
import sys
import types

import torch

from scripts import postprocess_scoremodel_ext_earthquake_run as postprocess


def _unit_points() -> torch.Tensor:
    return torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64
    )


def test_loads_ext_pt_without_upstream_coordinate_negation(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    generated = -_unit_points()
    train = _unit_points()
    validation = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    torch.save(generated, run_dir / "generated_samples.pt")
    torch.save(
        {
            "train_initial_points": train,
            "validation_initial_points": validation,
        },
        run_dir / "teacher_initial_points.pt",
    )

    observed, loaded_generated = postprocess.load_scoremodel_ext_run_artifacts(
        run_dir
    )

    torch.testing.assert_close(
        observed, torch.cat((train, validation)), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        loaded_generated, generated, rtol=0.0, atol=0.0
    )


def test_observed_points_fall_back_to_saved_teacher_datasets(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    train = _unit_points()
    validation = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    torch.save(-_unit_points(), run_dir / "generated_samples.pt")
    torch.save({"initial_point": train}, run_dir / "teacher_dataset.pt")
    torch.save(
        {"initial_point": validation}, run_dir / "validation_dataset.pt"
    )

    observed, _ = postprocess.load_scoremodel_ext_run_artifacts(run_dir)

    torch.testing.assert_close(
        observed, torch.cat((train, validation)), rtol=0.0, atol=0.0
    )


def test_single_run_postprocess_uses_shared_visualization_outputs(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    points = _unit_points()
    torch.save(points, run_dir / "generated_samples.pt")
    torch.save(
        {"train_initial_points": points},
        run_dir / "teacher_initial_points.pt",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "teacher": "heat",
                "score_parameterization": "upstream_scaled_score",
            }
        ),
        encoding="utf-8",
    )
    output_dir = run_dir / "viz"
    calls = []

    def save_pair(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
        pdf_path = path.with_suffix(".pdf")
        pdf_path.write_bytes(b"pdf")
        return path, pdf_path

    def fake_scatter(**kwargs):
        calls.append(("scatter", kwargs))
        output_path, pdf_path = save_pair(kwargs["output_path"])
        return {"output_path": output_path, "pdf_path": pdf_path}

    def fake_density(**kwargs):
        calls.append(("density", kwargs))
        global_path, global_pdf = save_pair(kwargs["global_output_path"])
        bandwidth_path, bandwidth_pdf = save_pair(
            kwargs["bandwidth_comparison_path"]
        )
        return {
            "global_output_path": global_path,
            "global_pdf_path": global_pdf,
            "bandwidth_comparison_path": bandwidth_path,
            "bandwidth_pdf_path": bandwidth_pdf,
        }

    fake_viz = types.ModuleType("scoremodel_ext.manifold.earthquake_smoke_viz")
    fake_viz.generate_earthquake_scatter_comparison = fake_scatter
    fake_viz.generate_earthquake_density_bandwidth_outputs = fake_density
    monkeypatch.setitem(
        sys.modules,
        "scoremodel_ext.manifold.earthquake_smoke_viz",
        fake_viz,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "postprocess_scoremodel_ext_earthquake_run.py",
            "--run-dir",
            str(run_dir),
        ],
    )

    postprocess.main()

    assert {path.name for path in output_dir.iterdir()} == {
        "scatter_global.png",
        "scatter_global.pdf",
        "scatter_japan_zoom.png",
        "scatter_japan_zoom.pdf",
        "density_global.png",
        "density_global.pdf",
        "density_bandwidth_comparison.png",
        "density_bandwidth_comparison.pdf",
    }
    assert [name for name, _ in calls] == ["scatter", "scatter", "density"]
    for _, kwargs in calls:
        assert kwargs["panel_order"] == ("observed", "generated")
        assert kwargs["panel_titles"]["generated"] == "Heat (Upstream-style)"
        torch.testing.assert_close(
            kwargs["generated_by_teacher"]["generated"],
            points,
            rtol=0.0,
            atol=0.0,
        )
    density_kwargs = calls[-1][1]
    assert density_kwargs["base_kappa"] == 80.0
    assert density_kwargs["bandwidth_scale"] == 0.5
