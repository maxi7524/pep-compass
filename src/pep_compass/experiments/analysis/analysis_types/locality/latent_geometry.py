"""Sequence and latent-space locality relationships."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pep_compass.experiments.analysis.result import AnalysisResult
from pep_compass.experiments.reader.selection import ExperimentSelection


def _parse_vector(value: object) -> np.ndarray | None:
    if not isinstance(value, str) or not value:
        return None
    vector = np.asarray(json.loads(value), dtype=float).reshape(-1)
    return vector if vector.size else None


def latent_locality(
    selection: ExperimentSelection,
    unique_per_run: bool = True,
    pogs_version: str | None = None,
    pogs_configuration: dict | None = None,
) -> AnalysisResult:
    """Relate edit distance to actual walker and encoded-candidate geometry."""
    occurrences = selection.collect(
        "candidate_occurrences",
        columns=[
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
        ],
    )
    candidates = selection.collect(
        "unique_candidates",
        columns=["run_id", "candidate_id", "sequence", "encoded_latent_mean"],
    )
    steps = selection.collect(
        "walker_steps",
        columns=[
            "run_id",
            "iteration_id",
            "trajectory_id",
            "step_id",
            "current_latent_position",
            "next_latent_position",
            "effective_dimension",
        ],
    )
    if occurrences.empty or candidates.empty or steps.empty:
        return AnalysisResult(pd.DataFrame(), {"analysis": "latent_locality"})
    keys = ["run_id", "iteration_id", "trajectory_id", "step_id"]
    frame = occurrences.merge(candidates, on=["run_id", "candidate_id"], how="inner")
    frame = frame.merge(steps, on=keys, how="inner")
    if unique_per_run:
        frame = frame.drop_duplicates(["run_id", "candidate_id"])
    candidate_vectors = frame["encoded_latent_mean"].map(_parse_vector)
    current_vectors = frame["current_latent_position"].map(_parse_vector)
    next_vectors = frame["next_latent_position"].map(_parse_vector)
    frame["walker_candidate_distance"] = [
        np.linalg.norm(candidate - current)
        if candidate is not None and current is not None and candidate.shape == current.shape
        else np.nan
        for candidate, current in zip(candidate_vectors, current_vectors)
    ]
    frame["next_walker_candidate_distance"] = [
        np.linalg.norm(candidate - next_position)
        if candidate is not None
        and next_position is not None
        and candidate.shape == next_position.shape
        else np.nan
        for candidate, next_position in zip(candidate_vectors, next_vectors)
    ]
    if pogs_version is not None:
        cached = selection.reader.metrics.get_metrics(
            "pogs",
            frame["sequence"],
            metric_version=pogs_version,
            configuration=pogs_configuration,
        )
        frame["pogs"] = frame["sequence"].map(cached)
    summary = (
        frame.groupby("levenshtein_to_center", dropna=False)
        .agg(
            candidate_count=("candidate_id", "nunique"),
            latent_distance_mean=("walker_candidate_distance", "mean"),
            latent_distance_median=("walker_candidate_distance", "median"),
            latent_distance_q25=(
                "walker_candidate_distance",
                lambda values: values.quantile(0.25),
            ),
            latent_distance_q75=(
                "walker_candidate_distance",
                lambda values: values.quantile(0.75),
            ),
        )
        .reset_index()
    )
    valid = frame[["levenshtein_to_center", "walker_candidate_distance"]].dropna()
    correlation = (
        spearmanr(
            valid["levenshtein_to_center"], valid["walker_candidate_distance"]
        )
        if len(valid) > 1
        and valid["levenshtein_to_center"].nunique() > 1
        and valid["walker_candidate_distance"].nunique() > 1
        else None
    )
    return AnalysisResult(
        summary,
        {
            "analysis": "latent_locality",
            "unique_per_run": unique_per_run,
            "spearman_correlation": (
                float(correlation.statistic) if correlation is not None else None
            ),
            "spearman_p_value": (
                float(correlation.pvalue) if correlation is not None else None
            ),
        },
        {"candidate_level_data": frame},
    )
