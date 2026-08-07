#!/usr/bin/env python3
"""Train the Ext Heat model with freshly generated teacher batches."""

from __future__ import annotations

import sys

if __package__:
    from .earthquake_heat_upstream_style_training import (
        configured_arguments as configured_fixed_arguments,
    )
    from .experiment_earthquake_teacher_compare_smoke import main as run_experiment
else:
    from earthquake_heat_upstream_style_training import (
        configured_arguments as configured_fixed_arguments,
    )
    from experiment_earthquake_teacher_compare_smoke import main as run_experiment


def _has_option(arguments: list[str], option: str) -> bool:
    return option in arguments or any(value.startswith(option + "=") for value in arguments)


def configured_arguments(arguments: list[str]) -> list[str]:
    configured = list(arguments)
    if _has_option(configured, "--teacher-sampling"):
        raise ValueError("this experiment fixes --teacher-sampling online")
    if not _has_option(configured, "--output-dir"):
        configured.extend(
            (
                "--output-dir",
                "results/earthquake_heat_ext_online_teacher",
            )
        )
    return ["--teacher-sampling", "online"] + configured_fixed_arguments(configured)


def main() -> None:
    sys.argv[1:] = configured_arguments(sys.argv[1:])
    run_experiment()


if __name__ == "__main__":
    main()
