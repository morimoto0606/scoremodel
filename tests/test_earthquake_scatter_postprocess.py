import ast
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import scoremodel_ext

from scoremodel_ext.manifold.earthquake_comparison_artifacts import (
    STANDARD_EARTH_COORDINATES,
    UPSTREAM_ANTIPODAL_COORDINATES,
    load_upstream_generated_samples,
)

from scripts.plot_earthquake_teacher_scatter_comparison import (
    parse_args,
    load_saved_scatter_artifacts,
)
from scripts.postprocess_earthquake_teacher_comparison import (
    build_metrics_comparison,
    comparison_panel_configuration,
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


def test_upstream_npy_loader_requires_explicit_coordinate_convention(tmp_path):
    path = tmp_path / "generated_samples.npy"
    np.save(path, np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]))

    with pytest.raises(ValueError, match="coordinate system is unknown"):
        load_upstream_generated_samples(path)

    converted = load_upstream_generated_samples(
        path,
        coordinate_system=UPSTREAM_ANTIPODAL_COORDINATES,
    )
    torch.testing.assert_close(
        converted,
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64),
        rtol=0,
        atol=0,
    )


def test_upstream_npy_loader_reads_standard_coordinate_metadata(tmp_path):
    path = tmp_path / "generated_samples.npy"
    np.save(path, np.array([[0.0, 0.0, 1.0]], dtype=np.float64))
    path.with_suffix(".json").write_text(
        json.dumps({"coordinate_system": STANDARD_EARTH_COORDINATES}),
        encoding="utf-8",
    )

    loaded = load_upstream_generated_samples(path)
    torch.testing.assert_close(
        loaded,
        torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64),
        rtol=0,
        atol=0,
    )


def test_upstream_panel_is_opt_in_and_preserves_legacy_order():
    legacy_order, _ = comparison_panel_configuration(
        {name: object() for name in ("heat", "varadhan", "malliavin")}
    )
    upstream_order, titles = comparison_panel_configuration(
        {
            name: object()
            for name in ("upstream", "heat", "varadhan", "malliavin")
        }
    )

    assert legacy_order == ("observed", "heat", "varadhan", "malliavin")
    assert upstream_order == (
        "observed",
        "upstream",
        "heat",
        "varadhan",
        "malliavin",
    )
    assert titles["upstream"] == "Upstream"


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
        assert kwargs["panel_order"] == (
            "observed",
            "heat",
            "varadhan",
            "malliavin",
        )
        calls.append(("scatter", kwargs.get("geographic_extent")))
        return save_pair(kwargs["output_path"])

    def fake_overlay(**kwargs):
        calls.append(("overlay", kwargs["geographic_extent"]))
        return save_pair(kwargs["output_path"])

    def fake_density(**kwargs):
        assert kwargs["panel_order"] == (
            "observed",
            "heat",
            "varadhan",
            "malliavin",
        )
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


def test_postprocess_passes_optional_upstream_panel_to_scatter_and_density(
    tmp_path,
    monkeypatch,
):
    upstream_path = tmp_path / "generated_samples.npy"
    np.save(
        upstream_path,
        np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64),
    )
    generated = {
        teacher: torch.ones(2, 3, dtype=torch.float64)
        for teacher in ("heat", "varadhan", "malliavin")
    }
    monkeypatch.setattr(
        postprocess,
        "load_saved_scatter_artifacts",
        lambda run_dirs: (torch.ones(2, 3, dtype=torch.float64), generated),
    )

    expected_order = (
        "observed",
        "upstream",
        "heat",
        "varadhan",
        "malliavin",
    )
    calls = []

    def fake_scatter(**kwargs):
        calls.append(("scatter", kwargs["panel_order"], tuple(kwargs["generated_by_teacher"])))
        return {"output_path": kwargs["output_path"], "pdf_path": None}

    def fake_density(**kwargs):
        calls.append(("density", kwargs["panel_order"], tuple(kwargs["generated_by_teacher"])))
        return {
            "global_output_path": kwargs["global_output_path"],
            "global_pdf_path": None,
            "bandwidth_comparison_path": kwargs["bandwidth_comparison_path"],
            "bandwidth_pdf_path": None,
        }

    def fake_overlay(**kwargs):
        return {"output_path": kwargs["output_path"], "pdf_path": None}

    fake_viz = types.ModuleType("scoremodel_ext.manifold.earthquake_smoke_viz")
    fake_viz.generate_earthquake_scatter_comparison = fake_scatter
    fake_viz.generate_earthquake_density_bandwidth_outputs = fake_density
    fake_viz.generate_earthquake_malliavin_overlay = fake_overlay
    monkeypatch.setitem(
        sys.modules,
        "scoremodel_ext.manifold.earthquake_smoke_viz",
        fake_viz,
    )
    monkeypatch.setattr(
        postprocess,
        "save_metrics_comparison",
        lambda run_dirs, output_path: {},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "postprocess_earthquake_teacher_comparison.py",
            "--upstream-samples",
            str(upstream_path),
            "--upstream-coordinate-system",
            UPSTREAM_ANTIPODAL_COORDINATES,
            "--comparison-dir",
            str(tmp_path / "comparison"),
        ],
    )

    postprocess.main()

    assert calls == [
        ("scatter", expected_order, ("upstream", "heat", "varadhan", "malliavin")),
        ("scatter", expected_order, ("upstream", "heat", "varadhan", "malliavin")),
        ("density", expected_order, ("upstream", "heat", "varadhan", "malliavin")),
    ]
