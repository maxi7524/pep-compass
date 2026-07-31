"""Isolated-process and Slurm execution backends for materialized plans."""

from __future__ import annotations

import shlex
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from pep_compass.experiments.plan import PlannedRun


def worker_command(
    config_path: Path,
    run_index: int | str,
    *,
    resume: bool = False,
    continue_on_error: bool = False,
) -> list[str]:
    """Return a command executing one plan entry in an isolated interpreter."""
    command = [
        sys.executable,
        "-m",
        "pep_compass.experiments.runner.composable_cli",
        "--config",
        str(config_path),
        "--run-index",
        str(run_index),
        "--worker",
    ]
    if resume:
        command.append("--resume")
    if continue_on_error:
        command.append("--continue-on-error")
    return command


def execute_subprocess_plan(
    plan: Sequence[PlannedRun],
    config_path: Path,
    *,
    max_workers: int = 1,
    resume: bool = False,
    continue_on_error: bool = False,
    command_runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> None:
    """Execute plan entries in isolated local Python processes."""
    if max_workers < 1:
        raise ValueError("Subprocess max_workers must be positive.")

    def execute(entry: PlannedRun) -> None:
        command_runner(
            worker_command(
                config_path,
                entry.index,
                resume=resume,
                continue_on_error=continue_on_error,
            ),
            check=True,
        )

    if max_workers == 1:
        for entry in plan:
            execute(entry)
        return
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(execute, plan))


def write_slurm_array_script(
    plan: Sequence[PlannedRun],
    config_path: Path,
    output_path: Path,
    *,
    settings: Mapping[str, Any] | None = None,
    resume: bool = False,
    continue_on_error: bool = False,
) -> Path:
    """Write, but do not submit, a Slurm array script for plan entries."""
    if not plan:
        raise ValueError("Cannot create a Slurm script for an empty plan.")
    indices = ",".join(str(entry.index) for entry in plan)
    options = dict(settings or {})
    directives = ["#!/usr/bin/env bash", f"#SBATCH --array={indices}"]
    for key in ("job_name", "partition", "time", "gres", "cpus_per_task", "mem"):
        value = options.get(key)
        if value is None:
            continue
        text = str(value)
        if "\n" in text or "\r" in text:
            raise ValueError(f"Invalid newline in Slurm setting: {key}")
        directive = key.replace("_", "-")
        directives.append(f"#SBATCH --{directive}={text}")
    command = worker_command(
        config_path,
        "$SLURM_ARRAY_TASK_ID",
        resume=resume,
        continue_on_error=continue_on_error,
    )
    rendered = [shlex.quote(part) for part in command]
    rendered[rendered.index("'$SLURM_ARRAY_TASK_ID'")] = '"$SLURM_ARRAY_TASK_ID"'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join([*directives, "", "set -euo pipefail", "", " ".join(rendered), ""]),
        encoding="utf-8",
    )
    output_path.chmod(0o755)
    return output_path
