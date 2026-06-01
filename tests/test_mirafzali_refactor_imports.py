import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scoremodel_ext.malliavin import experiment_mirafzali_nonlinear as exp_nl
from scoremodel_ext.malliavin import evaluation as eval_nl
from scoremodel_ext.malliavin import mirafzali_teacher as teacher_nl
from scoremodel_ext.malliavin import residual_correction as resid_nl


def test_old_imports_still_work_from_experiment_module():
    assert callable(exp_nl.run_experiment_nl)
    assert callable(exp_nl.build_training_dataset_nl)
    assert callable(exp_nl.compute_residuals_nl)
    assert callable(exp_nl.compute_metrics_nl)
    assert callable(exp_nl.build_results_table)
    assert hasattr(exp_nl, "ResidualCorrectionModel")


def test_new_module_imports_are_available():
    assert callable(teacher_nl.build_training_dataset_nl)
    assert hasattr(resid_nl, "ResidualCorrectionModel")
    assert callable(eval_nl.compute_metrics_nl)
    assert callable(eval_nl.build_results_table)


def test_experiment_reexports_match_new_module_symbols():
    assert exp_nl.build_training_dataset_nl is teacher_nl.build_training_dataset_nl
    assert exp_nl.ResidualCorrectionModel is resid_nl.ResidualCorrectionModel
    assert exp_nl.compute_metrics_nl is eval_nl.compute_metrics_nl
