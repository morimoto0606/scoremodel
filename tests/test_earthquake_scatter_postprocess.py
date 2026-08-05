import ast
import importlib
import json
import sys
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
        "results/earthquake_linear_beta_100k_ema_comparison/scatter_comparison.png"
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
