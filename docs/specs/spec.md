Refactor the Malliavin nonlinear experiment code into reusable modules.

Current problem:
`src/scoremodel_ext/malliavin/experiment_mirafzali_nonlinear.py` is too large and mixes:
- teacher construction
- residual correction models
- metrics
- plotting
- experiment runners
- sweep / multiseed evaluation

Goal:
Move reusable algorithmic code out of the experiment file, while preserving behavior and passing existing tests.

Do not change mathematical behavior in this refactor.

Files to create:
1. `src/scoremodel_ext/malliavin/mirafzali_teacher.py`
2. `src/scoremodel_ext/malliavin/residual_correction.py`
3. `src/scoremodel_ext/malliavin/evaluation.py`

Move the following from `experiment_mirafzali_nonlinear.py` to `mirafzali_teacher.py`:
- `_binned_score_at_points`
- `apply_teacher_nl`
- `simulate_all_times_nl`
- `build_training_dataset_nl`

Move the following to `residual_correction.py`:
- `_RESIDUAL_METHODS`
- `compute_residuals_nl`
- `_precompute_residual_bins`
- `_apply_bin_residual`
- `_nw_residual`
- `_knn_nw_residual`
- `ResidualCorrectionModel`
- `_plot_residual_field`

Move the following to `evaluation.py`:
- `compute_metrics_nl`
- `build_results_table`
- `_write_latex_table`

Then update `experiment_mirafzali_nonlinear.py` so that it imports these functions/classes from the new modules and keeps only:
- `run_experiment_nl`
- `run_residual_sweep`
- `run_phase_b`
- `run_mirafzali_baseline`
- `_MULTISEED_CONFIGS`
- `run_residual_multiseed_eval`
- `if __name__ == "__main__": ...`

Important constraints:
1. Preserve all public function names and signatures.
2. Preserve current behavior exactly.
3. Do not change numerical formulas.
4. Do not change default hyperparameters.
5. Do not rename result directories or metric keys.
6. Keep backwards compatibility for imports where reasonable.
7. Avoid circular imports.

Suggested dependencies:
- `mirafzali_teacher.py` may import:
  - `get_sampler`
  - `bin_teacher_2d`, `nw_teacher_2d`, `knn_nw_teacher_2d`
  - `simulate_malliavin_nl`
- `residual_correction.py` may import:
  - `knn_nw_teacher_2d`
  - matplotlib / numpy / torch
- `evaluation.py` may import:
  - `get_sampler`, `sample_8gmm`
  - `_mmd_rbf`, `_sliced_wasserstein`, `_mode_coverage_8gmm`
  - pandas, json, math, pathlib

After refactor:
- Run:
  `python -m py_compile src/scoremodel_ext/malliavin/*.py`
- Run:
  `python -m pytest tests/test_mirafzali_nonlinear.py -q`

Also add minimal tests if needed to verify:
- old imports still work if tests depend on them
- `run_experiment_nl` can import
- `ResidualCorrectionModel` can import from `residual_correction`
- `build_training_dataset_nl` can import from `mirafzali_teacher`

Do not implement new algorithms in this pass.
This is only a structural refactor.