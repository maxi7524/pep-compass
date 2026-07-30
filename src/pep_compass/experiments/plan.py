"""Materialized execution plan shared by local and external backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pep_compass.experiments.input import ExperimentTask, materialize_input_tasks
from pep_compass.experiments.variants import ExperimentVariant, materialize_variants


@dataclass(frozen=True)
class PlannedRun:
    """One stable entry in an experiment execution plan."""

    index: int
    variant: ExperimentVariant
    task: ExperimentTask
    seed: int | None


def materialize_execution_plan(
    config: Mapping[str, Any],
    *,
    base_directory: Path | None = None,
) -> tuple[PlannedRun, ...]:
    """Create deterministic ``variant × input task`` execution entries."""
    experiment = config.get("experiment", {})
    tasks = materialize_input_tasks(
        experiment.get("input", {}), base_directory=base_directory
    )
    variants = materialize_variants(config)
    base_seed = experiment.get("seed")
    entries = []
    for variant in variants:
        for task in tasks:
            index = variant.index * len(tasks) + task.index
            seed = base_seed + index if base_seed is not None else None
            entries.append(PlannedRun(index, variant, task, seed))
    return tuple(entries)
