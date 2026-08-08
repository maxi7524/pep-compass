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
    working_directory: Path | None = None,
) -> tuple[PlannedRun, ...]:
    """Create deterministic ``variant × input task`` execution entries.

    ``seed_scope=run`` assigns a distinct seed to every plan entry, whereas
    ``seed_scope=task`` reuses the input-task seed across grid variants.

    :param config: Complete experiment configuration.
    :param working_directory: Base directory for relative input paths.
    :return: Stable execution entries ordered by variant and input task.
    """
    experiment = config.get("experiment", {})
    tasks = materialize_input_tasks(
        experiment.get("input", {}),
        working_directory=working_directory or Path.cwd(),
    )
    variants = materialize_variants(config)
    base_seed = experiment.get("seed")
    seed_scope = experiment.get("seed_scope", "run")
    entries = []
    # Plan indices always remain globally unique, independently of seed scope.
    for variant in variants:
        for task in tasks:
            index = variant.index * len(tasks) + task.index
            seed_offset = task.index if seed_scope == "task" else index
            seed = base_seed + seed_offset if base_seed is not None else None
            entries.append(PlannedRun(index, variant, task, seed))
    return tuple(entries)
