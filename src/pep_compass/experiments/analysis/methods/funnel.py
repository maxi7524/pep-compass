"""Streaming summaries of candidate retention stages."""

from __future__ import annotations

import pandas as pd

from pep_compass.experiments.analysis.selection import ExperimentSelection


def retention_summary(
    selection: ExperimentSelection,
    group_by: list[str],
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    """Aggregate proposal and retention counts without loading all step rows.

    :param selection: Deferred experiment selection.
    :param group_by: Tracking or catalog columns defining result groups.
    :param chunk_size: Maximum source rows loaded at once.
    :return: Counts and stage-to-stage retention ratios per group.
    """
    if not group_by:
        raise ValueError("group_by must contain at least one column")
    count_columns = [
        "proposed_count",
        "post_limit_count",
        "post_method_filter_count",
        "post_constraint_filter_count",
    ]
    partials: list[pd.DataFrame] = []
    for chunk in selection.scan("steps", chunk_size=chunk_size):
        partials.append(chunk.groupby(group_by, dropna=False)[count_columns].sum())
    if not partials:
        return pd.DataFrame(columns=[*group_by, *count_columns])
    result = pd.concat(partials).groupby(level=list(range(len(group_by)))).sum()
    result["limit_retention"] = result["post_limit_count"].div(
        result["proposed_count"].replace(0, pd.NA)
    )
    result["method_retention"] = result["post_method_filter_count"].div(
        result["post_limit_count"].replace(0, pd.NA)
    )
    result["constraint_retention"] = result["post_constraint_filter_count"].div(
        result["post_method_filter_count"].replace(0, pd.NA)
    )
    return result.reset_index()
