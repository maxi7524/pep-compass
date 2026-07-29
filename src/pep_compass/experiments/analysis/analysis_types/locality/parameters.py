"""Partial multi-objective parameter summaries for locality experiments."""

from __future__ import annotations

import pandas as pd

from pep_compass.experiments.analysis.result import AnalysisResult
from pep_compass.experiments.reader.selection import ExperimentSelection


def parameter_selection(selection: ExperimentSelection) -> AnalysisResult:
    """Calculate preliminary yield and locality-retention objectives."""
    columns = [
        "run_id",
        "trajectory_count",
        "walker_step_count",
        "post_cap_event_count",
        "post_cap_unique_count",
        "post_constraint_event_count",
        "post_constraint_unique_count",
        "optimizer_pool_unique_count",
        "experiment",
        "grid_id",
        "method",
        "mutation.token_threshold",
    ]
    iterations = selection.collect("optimizer_iterations", columns=columns)
    if iterations.empty:
        return AnalysisResult(pd.DataFrame(), {"analysis": "parameter_selection"})
    numeric = [column for column in columns if column not in {
        "run_id", "experiment", "grid_id", "method"
    }]
    iterations[numeric] = iterations[numeric].apply(pd.to_numeric, errors="coerce")
    iterations["unique_candidates_per_trajectory"] = iterations[
        "post_constraint_unique_count"
    ].div(iterations["trajectory_count"].replace(0, pd.NA))
    iterations["unique_candidates_per_step"] = iterations[
        "post_constraint_unique_count"
    ].div(iterations["walker_step_count"].replace(0, pd.NA))
    iterations["constraint_retention"] = iterations[
        "post_constraint_unique_count"
    ].div(iterations["post_cap_unique_count"].replace(0, pd.NA))
    iterations["cross_trajectory_duplication"] = 1.0 - iterations[
        "post_cap_unique_count"
    ].div(iterations["post_cap_event_count"].replace(0, pd.NA))
    groups = ["experiment", "grid_id", "method", "mutation.token_threshold"]
    summary = iterations.groupby(groups, dropna=False).agg(
        run_count=("run_id", "nunique"),
        unique_candidates_mean=("post_constraint_unique_count", "mean"),
        candidates_per_trajectory_mean=("unique_candidates_per_trajectory", "mean"),
        candidates_per_step_mean=("unique_candidates_per_step", "mean"),
        constraint_retention_mean=("constraint_retention", "mean"),
        duplication_mean=("cross_trajectory_duplication", "mean"),
    ).reset_index()
    return AnalysisResult(
        summary,
        {
            "analysis": "parameter_selection",
            "status": "preliminary; APEX and PoGS objectives are not yet included",
        },
        {"iteration_level_data": iterations},
    )
