"""Partial multi-objective parameter summaries for locality experiments."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from pep_compass.experiments.analysis.analysis_types.locality._streaming import (
    ProgressReporter,
    Reservoir,
    RunningMoments,
)
from pep_compass.experiments.analysis.result import AnalysisResult
from pep_compass.experiments.reader.selection import ExperimentSelection


def parameter_selection(
    selection: ExperimentSelection,
    chunk_size: int = 100_000,
    diagnostic_sample_size: int = 10_000,
    random_seed: int = 0,
    progress: bool = True,
    progress_every_batches: int = 1,
) -> AnalysisResult:
    """Calculate preliminary objectives using partial chunk aggregates.

    :param selection: Runs and deferred row filters to analyze.
    :param chunk_size: Maximum optimizer-iteration rows loaded at once.
    :param diagnostic_sample_size: Maximum iteration-level rows returned in
        diagnostics.
    :param random_seed: Seed used for deterministic diagnostic sampling.
    :param progress: Print batch progress, throughput, and ETA.
    :param progress_every_batches: Number of batches between progress messages.
    :return: Grouped yield and locality-retention objectives.
    :raises ValueError: If a configured processing bound is not positive.
    """
    if min(chunk_size, diagnostic_sample_size, progress_every_batches) < 1:
        raise ValueError("Processing bounds must be positive")
    if progress:
        print("[parameter_selection] Counting optimizer iterations...", flush=True)
    total_rows = selection.count_rows("optimizer_iterations")
    reporter = ProgressReporter(
        "parameter_selection",
        total_rows,
        every_batches=progress_every_batches,
        enabled=progress,
    )
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
    numeric = [
        "trajectory_count",
        "walker_step_count",
        "post_cap_event_count",
        "post_cap_unique_count",
        "post_constraint_event_count",
        "post_constraint_unique_count",
        "optimizer_pool_unique_count",
        "mutation.token_threshold",
    ]
    groups = ["experiment", "grid_id", "method", "mutation.token_threshold"]
    metrics = [
        "post_constraint_unique_count",
        "unique_candidates_per_trajectory",
        "unique_candidates_per_step",
        "constraint_retention",
        "cross_trajectory_duplication",
    ]
    moments: dict[tuple[object, ...], dict[str, RunningMoments]] = defaultdict(
        lambda: {metric: RunningMoments() for metric in metrics}
    )
    run_ids: dict[tuple[object, ...], set[str]] = defaultdict(set)
    diagnostic_sample = Reservoir(diagnostic_sample_size, random_seed)
    diagnostic_columns: list[str] | None = None
    rows_scanned = 0

    for chunk in selection.scan(
        "optimizer_iterations", columns=columns, chunk_size=chunk_size
    ):
        rows_scanned += len(chunk)
        batch_rows = len(chunk)
        chunk[numeric] = chunk[numeric].apply(pd.to_numeric, errors="coerce")
        chunk["unique_candidates_per_trajectory"] = chunk[
            "post_constraint_unique_count"
        ].div(chunk["trajectory_count"].replace(0, np.nan))
        chunk["unique_candidates_per_step"] = chunk[
            "post_constraint_unique_count"
        ].div(chunk["walker_step_count"].replace(0, np.nan))
        chunk["constraint_retention"] = chunk[
            "post_constraint_unique_count"
        ].div(chunk["post_cap_unique_count"].replace(0, np.nan))
        chunk["cross_trajectory_duplication"] = 1.0 - chunk[
            "post_cap_unique_count"
        ].div(chunk["post_cap_event_count"].replace(0, np.nan))
        diagnostic_columns = chunk.columns.tolist()

        for raw_key, group in chunk.groupby(groups, dropna=False, sort=False):
            values = raw_key if isinstance(raw_key, tuple) else (raw_key,)
            key = tuple(None if pd.isna(value) else value for value in values)
            run_ids[key].update(group["run_id"].astype(str))
            for metric in metrics:
                moments[key][metric].update(group[metric].to_numpy(dtype=float))
        diagnostic_sample.update(chunk.itertuples(index=False, name=None))
        reporter.update(batch_rows)

    summary_rows = []
    for key, group_moments in moments.items():
        summary_rows.append(
            {
                **dict(zip(groups, key)),
                "run_count": len(run_ids[key]),
                "unique_candidates_mean": group_moments[
                    "post_constraint_unique_count"
                ].mean,
                "candidates_per_trajectory_mean": group_moments[
                    "unique_candidates_per_trajectory"
                ].mean,
                "candidates_per_step_mean": group_moments[
                    "unique_candidates_per_step"
                ].mean,
                "constraint_retention_mean": group_moments[
                    "constraint_retention"
                ].mean,
                "duplication_mean": group_moments[
                    "cross_trajectory_duplication"
                ].mean,
            }
        )
    return AnalysisResult(
        pd.DataFrame(summary_rows),
        {
            "analysis": "parameter_selection",
            "rows_scanned": rows_scanned,
            "elapsed_seconds": reporter.elapsed_seconds,
            "status": "preliminary; APEX and PoGS objectives are not yet included",
        },
        {
            "iteration_level_sample": pd.DataFrame(
                diagnostic_sample.values, columns=diagnostic_columns
            )
        },
    )
