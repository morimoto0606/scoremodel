from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

try:
    from .datasets_2d import get_sampler
    from .experiment_mirafzali import _mmd_rbf, _mode_coverage_8gmm, _sliced_wasserstein
except ImportError:
    from datasets_2d import get_sampler
    from experiment_mirafzali import _mmd_rbf, _mode_coverage_8gmm, _sliced_wasserstein


_METHOD_ORDER = [
    "raw", "binned", "nw", "knn_nw",
    "mirafzali",
    "mirafzali_residual_binned",
    "mirafzali_residual_nw",
    "mirafzali_residual_knn_nw",
]


def compute_metrics_nl(
    samples_np: np.ndarray,
    nan_rate: float,
    dataset_name: str,
    centers_np: Optional[np.ndarray] = None,
    n_ref: int = 10_000,
    rng=None,
) -> dict:
    """
    Compute MMD, sliced Wasserstein, and (for 8GMM) mode coverage.

    Reference samples are drawn fresh from the dataset distribution.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    sampler = get_sampler(dataset_name)
    result_r = sampler(n_ref, device="cpu")
    ref_np = (result_r[0] if isinstance(result_r, tuple) else result_r).numpy()

    metrics: dict = {
        "nan_rate": float(nan_rate),
        "n_samples": int(len(samples_np)),
        "mmd_rbf": _mmd_rbf(samples_np, ref_np, rng=rng),
        "sliced_wasserstein": _sliced_wasserstein(samples_np, ref_np, rng=rng),
    }

    if dataset_name == "8gmm" and centers_np is not None:
        metrics["mode_coverage"] = _mode_coverage_8gmm(samples_np, centers_np)

    return metrics


def build_results_table(
    results: dict,
    outbase: Optional[str] = None,
) -> "pd.DataFrame":
    """
    Flatten a nested results dict into a paper-style evaluation table.

    Parameters
    ----------
    results : dict
        Nested dict as returned by ``run_phase_b``:
        ``{dataset: {method: metrics_dict}}``.
    outbase : str or None
        If given, write ``summary.csv`` and ``summary.tex`` under this
        directory.  The directory is created if it does not exist.

    Returns
    -------
    pd.DataFrame
        Columns: dataset, method, mmd, sliced_wasserstein, nan_rate,
        and (for 8gmm rows only) coverage_fraction, mean_nearest_dist.
        Within each dataset block, the best (minimum) value in each
        numeric column is marked with a trailing ``*``.
    """
    base_cols = ["dataset", "method", "mmd", "sliced_wasserstein", "nan_rate"]
    gmm_cols = ["coverage_fraction", "mean_nearest_dist"]
    metric_cols = ["mmd", "sliced_wasserstein", "nan_rate"] + gmm_cols

    rows = []
    for dataset in sorted(results.keys()):
        method_dict = results[dataset]
        for method in _METHOD_ORDER:
            if method not in method_dict:
                continue
            m = method_dict[method]
            if "error" in m:
                continue
            row: dict = {
                "dataset": dataset,
                "method": method,
                "mmd": m.get("mmd_rbf"),
                "sliced_wasserstein": m.get("sliced_wasserstein"),
                "nan_rate": m.get("nan_rate"),
            }
            mc = m.get("mode_coverage", {})
            row["coverage_fraction"] = mc.get("coverage_fraction") if mc else None
            row["mean_nearest_dist"] = mc.get("mean_nearest_dist") if mc else None
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=base_cols + gmm_cols)

    df = pd.DataFrame(rows, columns=base_cols + gmm_cols)

    # Convert metric columns to object dtype so we can append "*" to strings
    # without triggering a dtype incompatibility warning in recent pandas.
    for col in metric_cols:
        df[col] = df[col].astype(object)

    # ── mark best value per dataset ───────────────────────────────────────
    # Lower is better for all metrics; mark minimum with "*".
    # We build a string version of the table for display / LaTeX export.
    for _, grp in df.groupby("dataset", sort=False):
        for col in metric_cols:
            numeric_vals = pd.to_numeric(grp[col], errors="coerce")
            if numeric_vals.notna().any():
                best_idx = numeric_vals.idxmin()
                df.loc[best_idx, col] = str(df.loc[best_idx, col]) + "*"

    # ── save ───────────────────────────────────────────────────────────────
    if outbase is not None:
        out = Path(outbase)
        out.mkdir(parents=True, exist_ok=True)

        csv_path = out / "summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Results table → {csv_path}")

        tex_path = out / "summary.tex"
        _write_latex_table(df, tex_path)
        print(f"  LaTeX table   → {tex_path}")

    return df


def _write_latex_table(df: "pd.DataFrame", path: Path) -> None:
    """
    Write a LaTeX booktabs table from the results DataFrame.

    Best values (marked with ``*``) are rendered in \\textbf{}.
    """
    col_labels = {
        "dataset": "Dataset",
        "method": "Method",
        "mmd": "MMD $\\downarrow$",
        "sliced_wasserstein": "SW $\\downarrow$",
        "nan_rate": "NaN rate $\\downarrow$",
        "coverage_fraction": "Cov.\\,frac.",
        "mean_nearest_dist": "Mean dist.",
    }

    def _fmt(val) -> str:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return "--"
        s = str(val)
        is_best = s.endswith("*")
        try:
            num = float(s.rstrip("*"))
            formatted = f"{num:.5f}"
        except ValueError:
            formatted = s.rstrip("*")
        return f"\\textbf{{{formatted}}}" if is_best else formatted

    visible_cols = [c for c in df.columns if c in col_labels]
    header = " & ".join(col_labels[c] for c in visible_cols) + " \\\\"

    lines = [
        "\\begin{table}[ht]",
        "  \\centering",
        "  \\caption{Nonlinear SDE — teacher comparison}",
        "  \\label{tab:nl_teacher_compare}",
        f"  \\begin{{tabular}}{{{'l' * len(visible_cols)}}}",
        "    \\toprule",
        f"    {header}",
        "    \\midrule",
    ]

    prev_ds = None
    for _, row in df.iterrows():
        if prev_ds is not None and row["dataset"] != prev_ds:
            lines.append("    \\midrule")
        cells = " & ".join(_fmt(row[c]) for c in visible_cols)
        lines.append(f"    {cells} \\\\")
        prev_ds = row["dataset"]

    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]

    path.write_text("\n".join(lines) + "\n")
