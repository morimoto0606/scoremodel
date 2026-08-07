#!/usr/bin/env python3
"""Train Earthquake Malliavin with Upstream's scaled-score parameterization.

The existing Malliavin teacher dataset is used unchanged.  Only the independent
direct-score training path is selected: the network predicts
``sigma(t) * s_Malliavin(x, t)`` and its public forward method divides by
``sigma(t)`` before the unchanged reverse sampler consumes the score.
"""

from __future__ import annotations

import sys

if __package__:
    from .experiment_earthquake_teacher_compare_smoke import main as run_experiment
else:
    from experiment_earthquake_teacher_compare_smoke import main as run_experiment


DEFAULT_ARGUMENTS = (
    ("--output-dir", "results/earthquake_malliavin_upstream_style_training"),
    ("--train-size", "4096"),
    ("--validation-size", "1024"),
    ("--n-steps", "32"),
    ("--teacher-implementation", "batched"),
    ("--teacher-batch-size", "4"),
    ("--minimum-time", "0.001"),
    ("--maximum-time", "1.0"),
    ("--time-sampling", "uniform"),
    ("--beta-schedule", "linear"),
    ("--beta-0", "0.001"),
    ("--beta-f", "5.0"),
    ("--beta-t0", "0.0"),
    ("--beta-tf", "1.0"),
    ("--epochs", "3000"),
    ("--training-unit", "updates"),
    ("--updates", "100000"),
    ("--warmup-updates", "1000"),
    ("--lr-scheduler", "cosine"),
    ("--ema-rate", "0.999"),
    ("--checkpoint-every-updates", "25000"),
    ("--batch-size", "512"),
    ("--learning-rate", "2e-4"),
    ("--weight-decay", "0"),
    ("--hidden", "1024"),
    ("--n-blocks", "6"),
    ("--num-frequencies", "16"),
    ("--reverse-steps", "256"),
    ("--n-generated-samples", "4096"),
    ("--dtype", "float64"),
)
DEFAULT_FLAGS = ("--use-ema-for-validation", "--use-ema-for-reverse", "--skip-viz")


def _has_option(arguments: list[str], option: str) -> bool:
    return option in arguments or any(
        value.startswith(option + "=") for value in arguments
    )


def configured_arguments(arguments: list[str]) -> list[str]:
    configured = list(arguments)
    if _has_option(configured, "--teacher"):
        raise ValueError("this experiment fixes --teacher malliavin")
    if _has_option(configured, "--score-parameterization"):
        raise ValueError(
            "this experiment fixes --score-parameterization upstream_scaled_score"
        )
    fixed = [
        "--teacher",
        "malliavin",
        "--score-parameterization",
        "upstream_scaled_score",
    ]
    defaults: list[str] = []
    for option, value in DEFAULT_ARGUMENTS:
        if not _has_option(configured, option):
            defaults.extend((option, value))
    for flag in DEFAULT_FLAGS:
        if not _has_option(configured, flag):
            defaults.append(flag)
    return fixed + defaults + configured


def main() -> None:
    sys.argv[1:] = configured_arguments(sys.argv[1:])
    run_experiment()


if __name__ == "__main__":
    main()
