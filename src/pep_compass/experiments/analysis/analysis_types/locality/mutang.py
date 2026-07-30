"""MUTANG selectivity and dimensionality summaries."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pep_compass.experiments.analysis.analysis_types.locality._streaming import (
    ProgressReporter,
    Reservoir,
    RunningMoments,
)
from pep_compass.experiments.analysis.result import AnalysisResult
from pep_compass.experiments.reader.selection import ExperimentSelection


def _group_key(values: tuple[object, ...]) -> tuple[object, ...]:
    """Normalize missing grouping values into stable dictionary keys."""
    return tuple(None if pd.isna(value) else value for value in values)


def mutang_selectivity(
    selection: ExperimentSelection,
    chunk_size: int = 100_000,
    distribution_sample_size: int = 50_000,
    diagnostic_sample_size: int = 10_000,
    random_seed: int = 0,
    progress: bool = True,
    progress_every_batches: int = 1,
) -> AnalysisResult:
    """Summarize MUTANG retention using partial chunk aggregates.

    :param selection: Runs and deferred row filters to analyze.
    :param chunk_size: Maximum walker-step rows loaded at once.
    :param distribution_sample_size: Maximum observations retained for each
        non-additive statistic and correlation.
    :param diagnostic_sample_size: Maximum step-level rows returned in
        diagnostics.
    :param random_seed: Seed used for deterministic reservoir sampling.
    :param progress: Print batch progress, throughput, and ETA.
    :param progress_every_batches: Number of batches between progress messages.
    :return: Grouped MUTANG selectivity summary.
    :raises ValueError: If a configured sample bound is not positive.
    """
    if min(
        chunk_size,
        distribution_sample_size,
        diagnostic_sample_size,
        progress_every_batches,
    ) < 1:
        raise ValueError("Processing bounds must be positive")
    if progress:
        print("[mutang_selectivity] Counting walker steps...", flush=True)
    total_rows = selection.count_rows("walker_steps")
    reporter = ProgressReporter(
        "mutang_selectivity",
        total_rows,
        every_batches=progress_every_batches,
        enabled=progress,
    )
    columns = [
        "run_id",
        "effective_dimension",
        "mutang_position_count",
        "mutang_residue_option_count",
        "theoretical_product_count",
        "post_cap_unique_count",
        "post_method_filter_unique_count",
        "post_constraint_unique_count",
        "experiment",
        "grid_id",
        "method",
        "mutation.token_threshold",
    ]
    numeric = [
        "effective_dimension",
        "mutang_position_count",
        "mutang_residue_option_count",
        "theoretical_product_count",
        "post_cap_unique_count",
        "post_method_filter_unique_count",
        "post_constraint_unique_count",
        "mutation.token_threshold",
    ]
    groups = ["experiment", "grid_id", "method", "mutation.token_threshold"]
    metric_columns = [
        "effective_dimension",
        "mutang_position_count",
        "mutang_residue_option_count",
        "post_cap_unique_count",
        "post_constraint_unique_count",
        "cap_efficiency",
        "constraint_retention",
    ]
    moments: dict[tuple[object, ...], dict[str, RunningMoments]] = defaultdict(
        lambda: {column: RunningMoments() for column in metric_columns}
    )
    theoretical_samples: dict[tuple[object, ...], Reservoir] = {}
    correlation_columns = (
        "effective_dimension",
        "post_cap_unique_count",
        "post_constraint_unique_count",
        "constraint_retention",
    )
    correlation_samples = {
        column: Reservoir(distribution_sample_size, random_seed + index)
        for index, column in enumerate(correlation_columns)
    }
    diagnostic_sample = Reservoir(diagnostic_sample_size, random_seed + 20)
    diagnostic_columns: list[str] | None = None
    rows_scanned = 0

    for chunk in selection.scan("walker_steps", columns=columns, chunk_size=chunk_size):
        rows_scanned += len(chunk)
        batch_rows = len(chunk)
        chunk[numeric] = chunk[numeric].apply(pd.to_numeric, errors="coerce")
        chunk["cap_efficiency"] = chunk["post_cap_unique_count"].div(
            chunk["theoretical_product_count"].replace(0, np.nan)
        )
        chunk["constraint_retention"] = chunk[
            "post_constraint_unique_count"
        ].div(chunk["post_cap_unique_count"].replace(0, np.nan))
        diagnostic_columns = chunk.columns.tolist()

        for raw_key, group in chunk.groupby(groups, dropna=False, sort=False):
            key = _group_key(raw_key if isinstance(raw_key, tuple) else (raw_key,))
            for column in metric_columns:
                moments[key][column].update(group[column].to_numpy(dtype=float))
            sample = theoretical_samples.setdefault(
                key,
                Reservoir(
                    distribution_sample_size,
                    random_seed + len(theoretical_samples) + 50,
                ),
            )
            values = group["theoretical_product_count"].to_numpy(dtype=float)
            sample.update(values[np.isfinite(values)].tolist())

        for column, reservoir in correlation_samples.items():
            valid = chunk[["mutation.token_threshold", column]].dropna()
            reservoir.update(valid.itertuples(index=False, name=None))
        diagnostic_sample.update(chunk.itertuples(index=False, name=None))
        reporter.update(batch_rows)

    summary_rows = []
    for key, group_moments in moments.items():
        theoretical = np.asarray(theoretical_samples[key].values, dtype=float)
        summary_rows.append(
            {
                **dict(zip(groups, key)),
                "step_count": group_moments["effective_dimension"].count,
                "effective_dimension_mean": group_moments[
                    "effective_dimension"
                ].mean,
                "mutation_positions_mean": group_moments[
                    "mutang_position_count"
                ].mean,
                "residue_options_mean": group_moments[
                    "mutang_residue_option_count"
                ].mean,
                "theoretical_product_median": float(np.median(theoretical)),
                "post_cap_unique_mean": group_moments[
                    "post_cap_unique_count"
                ].mean,
                "post_constraint_unique_mean": group_moments[
                    "post_constraint_unique_count"
                ].mean,
                "cap_efficiency_mean": group_moments["cap_efficiency"].mean,
                "constraint_retention_mean": group_moments[
                    "constraint_retention"
                ].mean,
            }
        )

    correlations = {}
    for column, reservoir in correlation_samples.items():
        sample = pd.DataFrame(reservoir.values, columns=["threshold", "value"])
        statistic = None
        if (
            len(sample) > 1
            and sample["threshold"].nunique() > 1
            and sample["value"].nunique() > 1
        ):
            statistic = spearmanr(sample["threshold"], sample["value"])
        correlations[column] = {
            "correlation": float(statistic.statistic) if statistic else None,
            "p_value": float(statistic.pvalue) if statistic else None,
            "sample_size": len(sample),
        }
    return AnalysisResult(
        pd.DataFrame(summary_rows),
        {
            "analysis": "mutang_selectivity",
            "rows_scanned": rows_scanned,
            "elapsed_seconds": reporter.elapsed_seconds,
            "distribution_statistics_are_sampled": True,
            "distribution_sample_size": distribution_sample_size,
            "correlations": correlations,
        },
        {
            "step_level_sample": pd.DataFrame(
                diagnostic_sample.values, columns=diagnostic_columns
            )
        },
    )
