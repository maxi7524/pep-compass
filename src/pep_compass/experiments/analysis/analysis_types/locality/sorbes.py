"""Trajectory-count and step-depth saturation analyses for SORBES."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from pep_compass.experiments.analysis.resampling import BootstrapEngine, RowSampler
from pep_compass.experiments.analysis.result import AnalysisResult
from pep_compass.experiments.reader.selection import ExperimentSelection


def _as_boolean(values: pd.Series) -> pd.Series:
    return values.astype("string").str.lower().map(
        {"true": True, "1": True, "false": False, "0": False}
    ).fillna(False)


def _candidate_sets(
    frame: pd.DataFrame, group_columns: list[str]
) -> pd.DataFrame:
    return (
        frame.groupby(group_columns, dropna=False)["candidate_id"]
        .agg(lambda values: frozenset(values))
        .rename("candidate_ids")
        .reset_index()
    )


def _unique_union_count(sample: pd.DataFrame) -> float:
    candidates: set[str] = set()
    for values in sample["candidate_ids"]:
        candidates.update(values)
    return float(len(candidates))


def trajectory_saturation(
    selection: ExperimentSelection,
    trajectory_counts: Sequence[int] | None = None,
    repetitions: int = 300,
    confidence: float = 0.95,
    random_seed: int = 0,
    accepted_only: bool = True,
) -> AnalysisResult:
    """Estimate candidate saturation under random trajectory subsets."""
    columns = [
        "run_id",
        "iteration_id",
        "trajectory_id",
        "candidate_id",
        "passed_constraint_filter",
        "experiment",
        "grid_id",
        "method",
        "mutation.token_threshold",
    ]
    occurrences = selection.collect("candidate_occurrences", columns=columns)
    if occurrences.empty:
        return AnalysisResult(pd.DataFrame(), {"analysis": "trajectory_saturation"})
    if accepted_only:
        occurrences = occurrences[_as_boolean(occurrences["passed_constraint_filter"])]
    sets = _candidate_sets(
        occurrences,
        [
            "run_id",
            "iteration_id",
            "trajectory_id",
            "experiment",
            "grid_id",
            "method",
            "mutation.token_threshold",
        ],
    )
    engine = BootstrapEngine()
    rows = []
    defaults = (1, 2, 5, 10, 20, 50, 100)
    for run_id, run in sets.groupby("run_id", sort=False):
        maximum = len(run)
        counts = trajectory_counts or defaults
        counts = sorted({int(value) for value in counts if 0 < value <= maximum})
        if maximum not in counts:
            counts.append(maximum)
        metadata = run.iloc[0]
        for count in counts:
            samples = engine.run(
                run,
                _unique_union_count,
                RowSampler(sample_size=count, replace=False),
                repetitions=repetitions,
                confidence=confidence,
                random_seed=random_seed + count,
            )
            rows.append(
                {
                    "run_id": run_id,
                    "experiment": metadata["experiment"],
                    "grid_id": metadata["grid_id"],
                    "method": metadata["method"],
                    "mutation.token_threshold": metadata[
                        "mutation.token_threshold"
                    ],
                    "trajectory_count": count,
                    "available_trajectories": maximum,
                    "unique_candidates_mean": samples["estimate"].iloc[0],
                    "confidence_low": samples["confidence_low"].iloc[0],
                    "confidence_high": samples["confidence_high"].iloc[0],
                }
            )
    result = pd.DataFrame(rows).sort_values(["run_id", "trajectory_count"])
    result["marginal_unique_candidates"] = result.groupby("run_id")[
        "unique_candidates_mean"
    ].diff()
    result["marginal_per_trajectory"] = result["marginal_unique_candidates"].div(
        result.groupby("run_id")["trajectory_count"].diff()
    )
    return AnalysisResult(
        result.reset_index(drop=True),
        {
            "analysis": "trajectory_saturation",
            "resampling": "random subsets without replacement",
            "accepted_only": accepted_only,
        },
    )


def step_saturation(
    selection: ExperimentSelection,
    max_steps: Sequence[int] | None = None,
    repetitions: int = 300,
    confidence: float = 0.95,
    random_seed: int = 0,
    accepted_only: bool = True,
) -> AnalysisResult:
    """Estimate cumulative candidate yield as trajectory depth increases."""
    columns = [
        "run_id",
        "iteration_id",
        "trajectory_id",
        "step_id",
        "candidate_id",
        "passed_constraint_filter",
        "experiment",
        "grid_id",
        "method",
        "mutation.token_threshold",
    ]
    occurrences = selection.collect("candidate_occurrences", columns=columns)
    if occurrences.empty:
        return AnalysisResult(pd.DataFrame(), {"analysis": "step_saturation"})
    if accepted_only:
        occurrences = occurrences[_as_boolean(occurrences["passed_constraint_filter"])]
    engine = BootstrapEngine()
    rows = []
    for run_id, run in occurrences.groupby("run_id", sort=False):
        available_steps = sorted(run["step_id"].dropna().astype(int).unique())
        selected_steps = sorted(
            set(int(value) for value in (max_steps or available_steps))
            & set(available_steps)
        )
        metadata = run.iloc[0]
        for max_step in selected_steps:
            truncated = run[run["step_id"] <= max_step]
            sets = _candidate_sets(
                truncated, ["iteration_id", "trajectory_id"]
            )
            samples = engine.run(
                sets,
                _unique_union_count,
                RowSampler(sample_size=len(sets), replace=True),
                repetitions=repetitions,
                confidence=confidence,
                random_seed=random_seed + max_step,
            )
            rows.append(
                {
                    "run_id": run_id,
                    "experiment": metadata["experiment"],
                    "grid_id": metadata["grid_id"],
                    "method": metadata["method"],
                    "mutation.token_threshold": metadata[
                        "mutation.token_threshold"
                    ],
                    "max_step": max_step,
                    "trajectory_count": len(sets),
                    "unique_candidates_mean": samples["estimate"].iloc[0],
                    "confidence_low": samples["confidence_low"].iloc[0],
                    "confidence_high": samples["confidence_high"].iloc[0],
                }
            )
    result = pd.DataFrame(rows).sort_values(["run_id", "max_step"])
    result["marginal_unique_candidates"] = result.groupby("run_id")[
        "unique_candidates_mean"
    ].diff()
    return AnalysisResult(
        result.reset_index(drop=True),
        {
            "analysis": "step_saturation",
            "resampling": "trajectory cluster bootstrap",
            "accepted_only": accepted_only,
        },
    )
