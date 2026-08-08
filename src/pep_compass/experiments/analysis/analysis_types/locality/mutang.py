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
from pep_compass.experiments.analysis.analysis_types.locality.sorbes import (
    CardinalitySketch,
    _as_boolean,
)
from pep_compass.experiments.analysis.result import AnalysisResult
from pep_compass.experiments.reader.selection import ExperimentSelection


def _group_key(values: tuple[object, ...]) -> tuple[object, ...]:
    """Normalize missing grouping values into stable dictionary keys."""
    return tuple(None if pd.isna(value) else value for value in values)


def _method_variant(metadata: dict[str, object]) -> str:
    """Return a compact label for one method-specific parameter variant."""
    method = str(metadata.get("method"))
    token = metadata.get("mutation.token_threshold")
    suffix = f", token={token}"
    if method == "lams":
        return f"LAMS sim={metadata.get('filter.similarity_threshold')}{suffix}"
    if method in {"tandem", "move", "lpbebo"}:
        return (
            f"{method.upper()} top-p={metadata.get('filter.top_p')}, "
            f"T={metadata.get('filter.temperature')}{suffix}"
        )
    if method == "random_mutang":
        return (
            "random MUTANG "
            f"fraction={metadata.get('filter.selection_fraction')}, "
            f"T={metadata.get('filter.temperature')}{suffix}"
        )
    if method == "random_walker":
        return (
            "random walker "
            f"positions={metadata.get('filter.maximum_positions')}{suffix}"
        )
    return f"{method}{suffix}"


def method_comparison(
    selection: ExperimentSelection,
    levenshtein_radii: tuple[int, ...] = tuple(range(1, 9)),
    sketch_precision: int = 12,
    chunk_size: int = 100_000,
    progress: bool = True,
    progress_every_batches: int = 10,
) -> AnalysisResult:
    """Compare method filtering per peptide and retrospective locality radius.

    :param selection: Runs and deferred row filters to analyze.
    :param levenshtein_radii: Cumulative Levenshtein radii plotted separately.
    :param sketch_precision: HyperLogLog precision controlling count error/RAM.
    :param chunk_size: Maximum candidate-occurrence rows loaded at once.
    :param progress: Print scan progress, throughput, and ETA.
    :param progress_every_batches: Batches between progress messages.
    :return: One row per run and radius with pre/post-method unique counts.
    """
    radii = tuple(sorted({int(value) for value in levenshtein_radii if value > 0}))
    if not radii:
        raise ValueError("At least one positive Levenshtein radius is required")
    if chunk_size < 1 or progress_every_batches < 1:
        raise ValueError("Processing bounds must be positive")

    rows: list[dict[str, object]] = []
    maximum_radius = max(radii)
    # Per-run streaming cardinality collection
    for run in selection.runs:
        run_selection = ExperimentSelection(
            selection.reader, (run,), selection.row_filters
        )
        total = run_selection.count_rows("candidate_occurrences")
        reporter = ProgressReporter(
            f"method_comparison:{run.run_id}",
            total,
            every_batches=progress_every_batches,
            enabled=progress,
        )
        exact = {
            stage: {
                distance: CardinalitySketch(sketch_precision)
                for distance in range(maximum_radius + 1)
            }
            for stage in ("post_cap", "post_method")
        }
        for chunk in run_selection.scan(
            "candidate_occurrences",
            columns=[
                "candidate_id",
                "passed_method_filter",
                "levenshtein_to_center",
            ],
            chunk_size=chunk_size,
        ):
            batch_rows = len(chunk)
            chunk["levenshtein_to_center"] = pd.to_numeric(
                chunk["levenshtein_to_center"], errors="coerce"
            )
            chunk = chunk[
                chunk["levenshtein_to_center"].between(0, maximum_radius)
            ]
            if chunk.empty:
                reporter.update(batch_rows)
                continue
            chunk["levenshtein_to_center"] = chunk[
                "levenshtein_to_center"
            ].astype(int)
            passed_method = _as_boolean(chunk["passed_method_filter"])
            for distance, group in chunk.groupby(
                "levenshtein_to_center", sort=False, observed=True
            ):
                exact["post_cap"][int(distance)].update(group["candidate_id"])
                method_group = group[passed_method.loc[group.index]]
                exact["post_method"][int(distance)].update(
                    method_group["candidate_id"]
                )
            reporter.update(batch_rows)

        # Cumulative locality summaries
        cumulative = {
            stage: CardinalitySketch(sketch_precision) for stage in exact
        }
        metadata = run.metadata
        previous_radius = -1
        for radius in radii:
            for distance in range(previous_radius + 1, radius + 1):
                for stage in cumulative:
                    cumulative[stage].merge(exact[stage][distance])
            post_cap = cumulative["post_cap"].estimate()
            post_method = cumulative["post_method"].estimate()
            rows.append(
                {
                    "run_id": run.run_id,
                    "name": metadata.get("name"),
                    "sequence": metadata.get("sequence"),
                    "experiment": metadata.get("experiment"),
                    "grid_id": metadata.get("grid_id"),
                    "method": metadata.get("method"),
                    "method_variant": _method_variant(metadata),
                    "levenshtein_radius": radius,
                    "post_cap_local_unique": post_cap,
                    "post_method_local_unique": post_method,
                    "method_retention_local": (
                        post_method / post_cap if post_cap else np.nan
                    ),
                    "mutation.token_threshold": metadata.get(
                        "mutation.token_threshold"
                    ),
                    "filter.similarity_threshold": metadata.get(
                        "filter.similarity_threshold"
                    ),
                    "filter.top_p": metadata.get("filter.top_p"),
                    "filter.temperature": metadata.get("filter.temperature"),
                    "filter.selection_fraction": metadata.get(
                        "filter.selection_fraction"
                    ),
                }
            )
            previous_radius = radius
    return AnalysisResult(
        pd.DataFrame(rows),
        {
            "analysis": "method_comparison",
            "unit": "peptide run",
            "radii_are_cumulative": True,
            "constraint_is_retrospective": True,
            "cardinality_method": "HyperLogLog",
            "cardinality_relative_standard_error": 1.04
            / np.sqrt(2**sketch_precision),
        },
    )


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
