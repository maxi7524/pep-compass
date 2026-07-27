"""Streaming summaries of method-filter scores and decisions."""

from __future__ import annotations

import pandas as pd

from pep_compass.experiments.analysis.selection import ExperimentSelection


def filter_score_summary(
    selection: ExperimentSelection,
    group_by: list[str],
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    """Summarize filter scores and acceptance without loading all candidates.

    :param selection: Deferred run and row selection.
    :param group_by: Catalog or tracking columns defining groups.
    :param chunk_size: Maximum rows held for one source chunk.
    :return: Counts, score moments, extrema, and method acceptance rate.
    :raises ValueError: If ``group_by`` is empty.
    """
    if not group_by:
        raise ValueError("group_by must contain at least one column")
    columns = [
        "method_score",
        "passed_method_filter",
        *[column for column in group_by if column not in selection.runs.columns],
    ]
    partials: list[pd.DataFrame] = []
    for chunk in selection.scan("candidates", columns=columns, chunk_size=chunk_size):
        chunk["method_score_squared"] = chunk["method_score"].pow(2)
        chunk["scored_count"] = chunk["method_score"].notna().astype(int)
        partials.append(
            chunk.groupby(group_by, dropna=False).agg(
                candidate_count=("passed_method_filter", "size"),
                accepted_count=("passed_method_filter", "sum"),
                scored_count=("scored_count", "sum"),
                score_sum=("method_score", "sum"),
                score_squared_sum=("method_score_squared", "sum"),
                score_min=("method_score", "min"),
                score_max=("method_score", "max"),
            )
        )
    if not partials:
        return pd.DataFrame()
    summed = pd.concat(partials)
    totals = summed.groupby(level=list(range(len(group_by))), dropna=False).agg(
        candidate_count=("candidate_count", "sum"),
        accepted_count=("accepted_count", "sum"),
        scored_count=("scored_count", "sum"),
        score_sum=("score_sum", "sum"),
        score_squared_sum=("score_squared_sum", "sum"),
        score_min=("score_min", "min"),
        score_max=("score_max", "max"),
    )
    totals["acceptance_rate"] = totals["accepted_count"] / totals["candidate_count"]
    totals["score_mean"] = totals["score_sum"] / totals["scored_count"].replace(0, pd.NA)
    totals["score_variance"] = (
        totals["score_squared_sum"] / totals["scored_count"].replace(0, pd.NA)
        - totals["score_mean"].pow(2)
    )
    return totals.reset_index()
