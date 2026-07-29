"""MUTANG selectivity and dimensionality summaries."""

from __future__ import annotations

import pandas as pd
from scipy.stats import spearmanr

from pep_compass.experiments.analysis.result import AnalysisResult
from pep_compass.experiments.reader.selection import ExperimentSelection


def mutang_selectivity(selection: ExperimentSelection) -> AnalysisResult:
    """Summarize candidate growth and retention against MUTANG thresholds."""
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
    steps = selection.collect("walker_steps", columns=columns)
    if steps.empty:
        return AnalysisResult(pd.DataFrame(), {"analysis": "mutang_selectivity"})
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
    steps[numeric] = steps[numeric].apply(pd.to_numeric, errors="coerce")
    steps["cap_efficiency"] = steps["post_cap_unique_count"].div(
        steps["theoretical_product_count"].replace(0, pd.NA)
    )
    steps["constraint_retention"] = steps["post_constraint_unique_count"].div(
        steps["post_cap_unique_count"].replace(0, pd.NA)
    )
    group_columns = [
        "experiment",
        "grid_id",
        "method",
        "mutation.token_threshold",
    ]
    summary = steps.groupby(group_columns, dropna=False).agg(
        step_count=("effective_dimension", "size"),
        effective_dimension_mean=("effective_dimension", "mean"),
        mutation_positions_mean=("mutang_position_count", "mean"),
        residue_options_mean=("mutang_residue_option_count", "mean"),
        theoretical_product_median=("theoretical_product_count", "median"),
        post_cap_unique_mean=("post_cap_unique_count", "mean"),
        post_constraint_unique_mean=("post_constraint_unique_count", "mean"),
        cap_efficiency_mean=("cap_efficiency", "mean"),
        constraint_retention_mean=("constraint_retention", "mean"),
    ).reset_index()
    correlations = {}
    for column in (
        "effective_dimension",
        "post_cap_unique_count",
        "post_constraint_unique_count",
        "constraint_retention",
    ):
        valid = steps[["mutation.token_threshold", column]].dropna()
        statistic = spearmanr(valid.iloc[:, 0], valid.iloc[:, 1]) if len(valid) > 1 else None
        correlations[column] = {
            "correlation": float(statistic.statistic) if statistic else None,
            "p_value": float(statistic.pvalue) if statistic else None,
        }
    return AnalysisResult(
        summary,
        {"analysis": "mutang_selectivity", "correlations": correlations},
        {"step_level_data": steps},
    )
