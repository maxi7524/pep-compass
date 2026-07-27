"""APEX and other black-box evaluation summaries."""

from __future__ import annotations

import pandas as pd

from pep_compass.experiments.analysis.selection import ExperimentSelection


def evaluation_summary(
    selection: ExperimentSelection,
    group_by: list[str],
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    """Aggregate evaluated objective values for selected experiment runs.

    :param selection: Deferred run and iteration selection.
    :param group_by: Catalog or evaluation columns defining result groups.
    :param chunk_size: Maximum evaluation rows held at once.
    :return: Evaluation counts, sum, mean, minimum, and maximum per group.
    """
    if not group_by:
        raise ValueError("group_by must contain at least one column")
    partials = []
    for chunk in selection.scan("evaluations", chunk_size=chunk_size):
        partials.append(
            chunk.groupby(group_by, dropna=False)["objective_value"].agg(
                ["count", "sum", "min", "max"]
            )
        )
    if not partials:
        return pd.DataFrame()
    combined = pd.concat(partials)
    result = combined.groupby(level=list(range(len(group_by))), dropna=False).agg(
        evaluation_count=("count", "sum"),
        objective_sum=("sum", "sum"),
        objective_min=("min", "min"),
        objective_max=("max", "max"),
    )
    result["objective_mean"] = result["objective_sum"] / result["evaluation_count"]
    return result.reset_index()
