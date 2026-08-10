"""Sequence and latent-space locality relationships."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pep_compass.analysis.analysis_types.locality._streaming import (
    ProgressReporter,
    Reservoir,
    RunningMoments,
)
from pep_compass.analysis.result import AnalysisResult
from pep_compass.analysis.reader.selection import ExperimentSelection


def _parse_vector(value: object) -> np.ndarray | None:
    """Parse one serialized latent vector, returning ``None`` when absent."""
    if isinstance(value, np.ndarray):
        return value.reshape(-1) if value.size else None
    if not isinstance(value, str) or not value:
        return None
    vector = np.asarray(json.loads(value), dtype=float).reshape(-1)
    return vector if vector.size else None


def _geometry_distances(
    candidates: pd.Series,
    current_positions: pd.Series,
    next_positions: pd.Series,
    origin_positions: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate row-aligned candidate and walker distances."""
    current_output = np.full(len(candidates), np.nan, dtype=float)
    next_output = np.full(len(candidates), np.nan, dtype=float)
    origin_output = np.full(len(candidates), np.nan, dtype=float)
    walker_drift_output = np.full(len(candidates), np.nan, dtype=float)
    for index, values in enumerate(
        zip(candidates, current_positions, next_positions, origin_positions)
    ):
        candidate_value, current_value, next_value, origin_value = values
        candidate = _parse_vector(candidate_value)
        current_position = _parse_vector(current_value)
        next_position = _parse_vector(next_value)
        origin_position = _parse_vector(origin_value)
        if (
            candidate is not None
            and current_position is not None
            and candidate.shape == current_position.shape
        ):
            current_output[index] = np.linalg.norm(candidate - current_position)
        if (
            candidate is not None
            and next_position is not None
            and candidate.shape == next_position.shape
        ):
            next_output[index] = np.linalg.norm(candidate - next_position)
        if (
            candidate is not None
            and origin_position is not None
            and candidate.shape == origin_position.shape
        ):
            origin_output[index] = np.linalg.norm(candidate - origin_position)
        if (
            current_position is not None
            and origin_position is not None
            and current_position.shape == origin_position.shape
        ):
            walker_drift_output[index] = np.linalg.norm(
                current_position - origin_position
            )
    return current_output, next_output, origin_output, walker_drift_output


def _allocate_sample(
    run_sizes: dict[str, int], sample_size: int
) -> dict[str, int]:
    """Allocate a global sample proportionally across non-empty runs."""
    total = sum(run_sizes.values())
    if total <= sample_size:
        return dict(run_sizes)
    exact = {
        run_id: sample_size * run_size / total
        for run_id, run_size in run_sizes.items()
    }
    allocated = {
        run_id: min(run_sizes[run_id], int(value))
        for run_id, value in exact.items()
    }
    remaining = sample_size - sum(allocated.values())
    order = sorted(
        run_sizes,
        key=lambda run_id: exact[run_id] - allocated[run_id],
        reverse=True,
    )
    for run_id in order:
        if remaining == 0:
            break
        if allocated[run_id] < run_sizes[run_id]:
            allocated[run_id] += 1
            remaining -= 1
    return allocated


def _sample_candidates(
    selection: ExperimentSelection,
    population_size: int,
    sample_size: int,
    chunk_size: int,
    random_seed: int,
    progress: bool,
    progress_every_batches: int,
) -> pd.DataFrame:
    """Load a uniform bounded sample from one run's candidate table."""
    reporter = ProgressReporter(
        f"latent_locality:candidate_sampling:{selection.runs[0].run_id}",
        population_size,
        every_batches=progress_every_batches,
        enabled=progress,
    )
    reservoir = Reservoir(sample_size, random_seed)
    columns = ["candidate_id", "sequence", "encoded_latent_mean"]
    for chunk in selection.scan(
        "unique_candidates", columns=columns, chunk_size=chunk_size
    ):
        reservoir.update(chunk.itertuples(index=False, name=None))
        reporter.update(len(chunk))
    return pd.DataFrame(reservoir.values, columns=columns)


def latent_locality(
    selection: ExperimentSelection,
    unique_per_run: bool = True,
    pogs_version: str | None = None,
    pogs_configuration: dict | None = None,
    chunk_size: int = 20_000,
    quantile_sample_size: int = 50_000,
    diagnostic_sample_size: int = 10_000,
    candidate_sample_size: int | None = 50_000,
    candidate_scan_chunk_size: int = 10_000,
    random_seed: int = 0,
    progress: bool = True,
    progress_every_batches: int = 1,
) -> AnalysisResult:
    """Relate edit distance to latent geometry with bounded working memory.

    Candidate occurrences are scanned in chunks. Only the run-local candidate
    dictionary, walker-step table, partial aggregates, and bounded reservoir
    samples remain resident in memory.

    :param selection: Runs and deferred row filters to analyze.
    :param unique_per_run: Analyze only the first occurrence of each candidate.
    :param pogs_version: Optional cached PoGS metric version to attach to the
        diagnostic sample.
    :param pogs_configuration: Configuration identifying cached PoGS values.
    :param chunk_size: Maximum candidate occurrences joined at once.
    :param quantile_sample_size: Maximum latent distances retained per edit
        distance for approximate quartiles.
    :param diagnostic_sample_size: Maximum candidate-level rows returned in
        diagnostics.
    :param candidate_sample_size: Global number of unique candidates sampled
        proportionally across runs. ``None`` analyzes the complete population.
    :param candidate_scan_chunk_size: Maximum unique-candidate rows held while
        constructing the representative sample.
    :param random_seed: Seed used for deterministic reservoir sampling.
    :param progress: Print batch progress, throughput, and ETA.
    :param progress_every_batches: Number of batches between progress messages.
    :return: Edit-distance summary and bounded candidate-level diagnostics.
    :raises ValueError: If any configured processing bound is not positive.
    """
    if min(
        chunk_size,
        quantile_sample_size,
        diagnostic_sample_size,
        candidate_scan_chunk_size,
        progress_every_batches,
    ) < 1:
        raise ValueError("Processing bounds must be positive")
    if candidate_sample_size is not None and candidate_sample_size < 1:
        raise ValueError("Candidate sample size must be positive or None")

    if progress:
        print("[latent_locality] Counting candidate occurrences...", flush=True)
    total_rows = selection.count_rows("candidate_occurrences")
    reporter = ProgressReporter(
        "latent_locality",
        total_rows,
        every_batches=progress_every_batches,
        enabled=progress,
    )

    if progress:
        print("[latent_locality] Counting unique candidates per run...", flush=True)
    run_candidate_counts = {}
    for run in selection.runs:
        count = ExperimentSelection(
            selection.reader, (run,), selection.row_filters
        ).count_rows("unique_candidates")
        run_candidate_counts[run.run_id] = count
        if progress:
            print(
                f"[latent_locality] {run.run_id}: {count:,} unique candidates",
                flush=True,
            )
    if candidate_sample_size is None:
        run_sample_sizes = dict(run_candidate_counts)
    else:
        run_sample_sizes = _allocate_sample(
            run_candidate_counts, candidate_sample_size
        )
    if progress:
        print(
            "[latent_locality] Candidate population: "
            f"{sum(run_candidate_counts.values()):,}; planned sample: "
            f"{sum(run_sample_sizes.values()):,}",
            flush=True,
        )

    metric_names = (
        "walker_candidate_distance",
        "origin_candidate_distance",
        "walker_origin_distance",
    )
    distance_moments = {
        name: defaultdict(RunningMoments) for name in metric_names
    }
    quantiles: dict[str, dict[object, Reservoir]] = {
        name: {} for name in metric_names
    }
    candidate_counts: dict[object, int] = defaultdict(int)
    correlation_sample = Reservoir(quantile_sample_size, random_seed + 1)
    diagnostic_sample = Reservoir(diagnostic_sample_size, random_seed + 2)
    counted_candidates: set[tuple[object, ...]] = set()
    analyzed_candidates: set[tuple[str, str]] = set()
    rows_scanned = 0
    rows_analyzed = 0
    diagnostic_columns: list[str] | None = None
    sampled_candidates = 0

    occurrence_columns = [
        "run_id",
        "iteration_id",
        "trajectory_id",
        "step_id",
        "candidate_id",
        "passed_method_filter",
        "passed_constraint_filter",
        "method_score",
        "levenshtein_to_center",
        "levenshtein_to_parent",
        "experiment",
        "grid_id",
        "method",
        "mutation.token_threshold",
    ]
    for run_index, run in enumerate(selection.runs):
        if progress:
            print(
                f"[latent_locality] Preparing run {run_index + 1}/"
                f"{len(selection.runs)} ({run.run_id}): loading lookup tables",
                flush=True,
            )
        run_selection = ExperimentSelection(
            selection.reader, (run,), selection.row_filters
        )
        run_sample_size = run_sample_sizes.get(run.run_id, 0)
        if run_sample_size == 0:
            continue
        if run_sample_size < run_candidate_counts[run.run_id]:
            candidates = _sample_candidates(
                run_selection,
                run_candidate_counts[run.run_id],
                run_sample_size,
                candidate_scan_chunk_size,
                random_seed + run_index + 100,
                progress,
                progress_every_batches,
            )
        else:
            candidates = run_selection.collect(
                "unique_candidates",
                columns=["candidate_id", "sequence", "encoded_latent_mean"],
                chunk_size=candidate_scan_chunk_size,
            )
        steps = run_selection.collect(
            "walker_steps",
            columns=[
                "iteration_id",
                "trajectory_id",
                "step_id",
                "current_latent_position",
                "next_latent_position",
                "effective_dimension",
            ],
            chunk_size=chunk_size,
        )
        if candidates.empty or steps.empty:
            if progress:
                print(
                    f"[latent_locality] Skipping {run.run_id}: candidate or "
                    "walker-step table is empty",
                    flush=True,
                )
            continue
        candidates = candidates.drop_duplicates("candidate_id").set_index(
            "candidate_id"
        )
        sampled_candidates += len(candidates)
        step_keys = ["iteration_id", "trajectory_id", "step_id"]
        steps = steps.drop_duplicates(step_keys)
        steps["current_latent_position"] = steps[
            "current_latent_position"
        ].map(_parse_vector)
        steps["next_latent_position"] = steps["next_latent_position"].map(
            _parse_vector
        )
        trajectory_keys = ["iteration_id", "trajectory_id"]
        origin_positions = (
            steps.sort_values("step_id")
            .drop_duplicates(trajectory_keys)
            .set_index(trajectory_keys)["current_latent_position"]
            .to_dict()
        )
        steps["origin_latent_position"] = [
            origin_positions[(iteration, trajectory)]
            for iteration, trajectory in steps[trajectory_keys].itertuples(
                index=False, name=None
            )
        ]
        steps = steps.set_index(step_keys)
        if progress:
            print(
                f"[latent_locality] Prepared {run.run_id}: "
                f"{len(candidates):,} unique candidates, "
                f"{len(steps):,} walker steps",
                flush=True,
            )

        for chunk in run_selection.scan(
            "candidate_occurrences",
            columns=occurrence_columns,
            chunk_size=chunk_size,
        ):
            rows_scanned += len(chunk)
            batch_rows = len(chunk)
            chunk = chunk[chunk["candidate_id"].isin(candidates.index)]
            if chunk.empty:
                reporter.update(batch_rows)
                continue
            chunk = chunk.join(candidates, on="candidate_id", how="inner")
            chunk = chunk.join(steps, on=step_keys, how="inner")
            if chunk.empty:
                reporter.update(batch_rows)
                continue
            if unique_per_run:
                fresh = []
                for candidate_id in chunk["candidate_id"]:
                    key = (run.run_id, str(candidate_id))
                    fresh.append(key not in analyzed_candidates)
                    analyzed_candidates.add(key)
                chunk = chunk.loc[fresh]
                if chunk.empty:
                    reporter.update(batch_rows)
                    continue
            rows_analyzed += len(chunk)
            (
                current_distances,
                next_distances,
                origin_distances,
                walker_drift_distances,
            ) = _geometry_distances(
                chunk["encoded_latent_mean"],
                chunk["current_latent_position"],
                chunk["next_latent_position"],
                chunk["origin_latent_position"],
            )
            chunk["walker_candidate_distance"] = current_distances
            chunk["next_walker_candidate_distance"] = next_distances
            chunk["origin_candidate_distance"] = origin_distances
            chunk["walker_origin_distance"] = walker_drift_distances
            diagnostic_columns = chunk.columns.tolist()

            for edit_distance, group in chunk.groupby(
                "levenshtein_to_center", dropna=False, sort=False
            ):
                group_key = None if pd.isna(edit_distance) else edit_distance
                for metric_index, metric_name in enumerate(metric_names):
                    values = group[metric_name].to_numpy(dtype=float)
                    distance_moments[metric_name][group_key].update(values)
                    metric_quantiles = quantiles[metric_name]
                    reservoir = metric_quantiles.setdefault(
                        group_key,
                        Reservoir(
                            quantile_sample_size,
                            random_seed
                            + run_index
                            + len(metric_quantiles)
                            + 100 * metric_index
                            + 10,
                        ),
                    )
                    reservoir.update(values[np.isfinite(values)].tolist())
                for candidate_id in group["candidate_id"]:
                    candidate_key = (
                        (run.run_id, str(candidate_id))
                        if unique_per_run
                        else (run.run_id, str(candidate_id), group_key)
                    )
                    if candidate_key not in counted_candidates:
                        counted_candidates.add(candidate_key)
                        candidate_counts[group_key] += 1

            valid = chunk[
                ["levenshtein_to_center", "walker_candidate_distance"]
            ].dropna()
            correlation_sample.update(valid.itertuples(index=False, name=None))
            diagnostic_sample.update(chunk.itertuples(index=False, name=None))
            reporter.update(batch_rows)

    summary_rows = []
    current_moments = distance_moments["walker_candidate_distance"]
    for edit_distance in current_moments:
        metric_values = {}
        for metric_name in metric_names:
            sample = np.asarray(
                quantiles[metric_name][edit_distance].values, dtype=float
            )
            if sample.size:
                median = float(np.quantile(sample, 0.5))
                q25 = float(np.quantile(sample, 0.25))
                q75 = float(np.quantile(sample, 0.75))
            else:
                median = q25 = q75 = float("nan")
            metric_values.update(
                {
                    f"{metric_name}_mean": distance_moments[metric_name][
                        edit_distance
                    ].mean,
                    f"{metric_name}_median": median,
                    f"{metric_name}_q25": q25,
                    f"{metric_name}_q75": q75,
                }
            )
        summary_rows.append(
            {
                "levenshtein_to_center": edit_distance,
                "candidate_count": candidate_counts[edit_distance],
                **metric_values,
            }
        )
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(
            "levenshtein_to_center", na_position="last"
        ).reset_index(drop=True)

    correlation_values = correlation_sample.values
    correlation = None
    if len(correlation_values) > 1:
        correlation_frame = pd.DataFrame(
            correlation_values, columns=["edit_distance", "latent_distance"]
        )
        if (
            correlation_frame["edit_distance"].nunique() > 1
            and correlation_frame["latent_distance"].nunique() > 1
        ):
            correlation = spearmanr(
                correlation_frame["edit_distance"],
                correlation_frame["latent_distance"],
            )

    diagnostic = pd.DataFrame(
        diagnostic_sample.values,
        columns=diagnostic_columns,
    )
    if pogs_version is not None and not diagnostic.empty:
        cached = selection.reader.metrics.get_metrics(
            "pogs",
            diagnostic["sequence"],
            metric_version=pogs_version,
            configuration=pogs_configuration,
        )
        diagnostic["pogs"] = diagnostic["sequence"].map(cached)
    return AnalysisResult(
        summary,
        {
            "analysis": "latent_locality",
            "unique_per_run": unique_per_run,
            "rows_scanned": rows_scanned,
            "rows_analyzed": rows_analyzed,
            "candidate_population_size": sum(run_candidate_counts.values()),
            "candidate_sample_size": sampled_candidates,
            "candidate_sampling_fraction": (
                sampled_candidates / sum(run_candidate_counts.values())
                if sum(run_candidate_counts.values())
                else None
            ),
            "candidate_sampling": (
                "uniform_without_replacement_proportional_by_run"
                if candidate_sample_size is not None
                else "complete_population"
            ),
            "elapsed_seconds": reporter.elapsed_seconds,
            "quantiles_are_sampled": True,
            "quantile_sample_size_per_group": quantile_sample_size,
            "spearman_sample_size": len(correlation_values),
            "spearman_correlation": (
                float(correlation.statistic) if correlation is not None else None
            ),
            "spearman_p_value": (
                float(correlation.pvalue) if correlation is not None else None
            ),
        },
        {"candidate_level_sample": diagnostic},
    )
