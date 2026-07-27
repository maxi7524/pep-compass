"""Reconstruction of normalized SORBES trajectory paths."""

from __future__ import annotations

import pandas as pd

from pep_compass.experiments.analysis.selection import ExperimentSelection


def reconstruct_trajectory(
    selection: ExperimentSelection,
    run_id: str,
    iteration_id: int,
    trajectory_id: int,
) -> pd.DataFrame:
    """Load and order one walker trajectory from normalized step records.

    :param selection: Deferred experiment selection containing the run.
    :param run_id: Task identifier written to tracking tables.
    :param iteration_id: Optimizer iteration to reconstruct.
    :param trajectory_id: SORBES walker trajectory identifier.
    :return: Ordered step records for exactly one trajectory.
    """
    narrowed = selection.rows(trajectory_ids=[trajectory_id])
    frame = narrowed.collect("steps")
    frame = frame[
        (frame["run_id"] == run_id) & (frame["iteration_id"] == iteration_id)
    ]
    return frame.sort_values("step_id").reset_index(drop=True)
