import ast
import importlib
import json
import sys
import types
from pathlib import Path

import pytest
import torch
import scoremodel_ext

from scripts.plot_earthquake_teacher_scatter_comparison import (
    parse_args,
    load_saved_scatter_artifacts,
)
from scripts.postprocess_earthquake_teacher_comparison import (
    build_metrics_comparison,
    save_metrics_comparison,
)
from scripts import postprocess_earthquake_teacher_comparison as postprocess


def test_scatter_postprocess_loads_skip_viz_artifacts(tmp_path):
    run_dirs = {}
    for index, teacher in enumerate(("heat", "varadhan", "malliavin")):
        run_dir = tmp_path / teacher
        run_dir.mkdir()
        run_dirs[teacher] = run_dir
        torch.save(
            torch.full((4, 3), float(index), dtype=torch.float64),
            run_dir / "generated_samples.pt",
        )

    train = torch.arange(15, dtype=torch.float64).reshape(5, 3)
    validation = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    torch.save(
        {
            "train_initial_points": train,
            "validation_initial_points": validation,
        },
        run_dirs["heat"] / "teacher_initial_points.pt",
    )

    observed, generated = load_saved_scatter_artifacts(run_dirs)

    torch.testing.assert_close(observed, torch.cat((train, validation)), rtol=0, atol=0)
    assert generated.keys() == {"heat", "varadhan", "malliavin"}
    for index, teacher in enumerate(("heat", "varadhan", "malliavin")):
        torch.testing.assert_close(
            generated[teacher],
            torch.full((4, 3), float(index), dtype=torch.float64),
            rtol=0,
            atol=0,
        )


def test_scatter_default_output_is_in_comparison_directory(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["plot_earthquake_teacher_scatter_comparison.py"])
    args = parse_args()
    assert args.output == Path(
        "results/earthquake_linear_beta_100k_ema_comparison/scatter_global.png"
    )


def test_metrics_comparison_uses_final_train_loss_fallback(tmp_path):
    run_dirs = {}
    for index, teacher in enumerate(("heat", "varadhan", "malliavin"), start=1):
        run_dir = tmp_path / teacher
        run_dir.mkdir()
        run_dirs[teacher] = run_dir
        payload = {
            "final_train_loss": index * 1.0,
            "validation_loss": index * 2.0,
            "s2_rbf_mmd": index * 3.0,
            "nearest_neighbor_geodesic_mean": index * 4.0,
            "nearest_neighbor_geodesic_median": index * 5.0,
            "reverse_sampling_seconds": index * 6.0,
        }
        (run_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")

    comparison = build_metrics_comparison(run_dirs)
    assert comparison["heat"]["train_loss"] == 1.0
    assert comparison["malliavin"]["reverse_sampling_seconds"] == 18.0

    output_path = tmp_path / "comparison" / "metrics_comparison.json"
    saved = save_metrics_comparison(run_dirs, output_path)
    assert json.loads(output_path.read_text(encoding="utf-8")) == saved


def test_scoremodel_ext_does_not_import_scripts_package():
    package_root = Path(scoremodel_ext.__file__).resolve().parent
    violations = []
    for source_path in package_root.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module == "scripts" or node.module.startswith("scripts."):
                    violations.append(f"{source_path}:{node.lineno}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "scripts" or alias.name.startswith("scripts."):
                        violations.append(f"{source_path}:{node.lineno}")
    assert violations == []


def test_earthquake_smoke_viz_imports_without_scripts_package():
    pytest.importorskip("matplotlib.pyplot")
    pytest.importorskip("cartopy.crs")
    module = importlib.import_module("scoremodel_ext.manifold.earthquake_smoke_viz")
    assert hasattr(module, "generate_earthquake_density_comparison")


def test_postprocess_generates_global_zoom_density_and_overlay_paths(
    tmp_path,
    monkeypatch,
):
    run_dirs = {}
    for index, teacher in enumerate(("heat", "varadhan", "malliavin"), start=1):
        run_dir = tmp_path / teacher
        run_dir.mkdir()
        run_dirs[teacher] = run_dir
        torch.save(
            torch.full((4, 3), float(index), dtype=torch.float64),
            run_dir / "generated_samples.pt",
        )
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "final_train_loss": 1.0,
                    "validation_loss": 2.0,
                    "s2_rbf_mmd": 3.0,
                    "nearest_neighbor_geodesic_mean": 4.0,
                    "nearest_neighbor_geodesic_median": 5.0,
                    "reverse_sampling_seconds": 6.0,
                }
            ),
            encoding="utf-8",
        )
    torch.save(
        {"train_initial_points": torch.ones(4, 3, dtype=torch.float64)},
        run_dirs["heat"] / "teacher_initial_points.pt",
    )

    calls = []

    def save_pair(path):
        path.write_bytes(b"png")
        pdf_path = path.with_suffix(".pdf")
        pdf_path.write_bytes(b"pdf")
        return {"output_path": path, "pdf_path": pdf_path}

    def fake_scatter(**kwargs):
        calls.append(("scatter", kwargs.get("geographic_extent")))
        return save_pair(kwargs["output_path"])

    def fake_overlay(**kwargs):
        calls.append(("overlay", kwargs["geographic_extent"]))
        return save_pair(kwargs["output_path"])

    def fake_density(**kwargs):
        global_result = save_pair(kwargs["global_output_path"])
        bandwidth_result = save_pair(kwargs["bandwidth_comparison_path"])
        return {
            "global_output_path": global_result["output_path"],
            "global_pdf_path": global_result["pdf_path"],
            "bandwidth_comparison_path": bandwidth_result["output_path"],
            "bandwidth_pdf_path": bandwidth_result["pdf_path"],
        }

    fake_viz = types.ModuleType("scoremodel_ext.manifold.earthquake_smoke_viz")
    fake_viz.generate_earthquake_scatter_comparison = fake_scatter
    fake_viz.generate_earthquake_malliavin_overlay = fake_overlay
    fake_viz.generate_earthquake_density_bandwidth_outputs = fake_density
    monkeypatch.setitem(
        sys.modules,
        "scoremodel_ext.manifold.earthquake_smoke_viz",
        fake_viz,
    )
    comparison_dir = tmp_path / "comparison"
    argv = ["postprocess_earthquake_teacher_comparison.py"]
    for teacher in ("heat", "varadhan", "malliavin"):
        argv.extend((f"--{teacher}-dir", str(run_dirs[teacher])))
    argv.extend(("--comparison-dir", str(comparison_dir)))
    monkeypatch.setattr(sys, "argv", argv)

    postprocess.main()

    expected = {
        "scatter_global.png",
        "scatter_global.pdf",
        "scatter_japan_zoom.png",
        "scatter_japan_zoom.pdf",
        "density_global.png",
        "density_global.pdf",
        "density_bandwidth_comparison.png",
        "density_bandwidth_comparison.pdf",
        "malliavin_overlay_japan_zoom.png",
        "malliavin_overlay_japan_zoom.pdf",
        "metrics_comparison.json",
    }
    assert {path.name for path in comparison_dir.iterdir()} == expected
    japan_extent = (120.0, 150.0, 20.0, 50.0)
    assert calls == [
        ("scatter", None),
        ("scatter", japan_extent),
        ("overlay", japan_extent),
    ]
