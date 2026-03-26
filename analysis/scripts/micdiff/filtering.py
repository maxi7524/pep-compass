"""Filtering functionality for parent and mutant dataframes."""

import pandas as pd
from loguru import logger


def filter_parents_and_mutants(
    parents_df: pd.DataFrame,
    mutants_df: pd.DataFrame,
    filter_name: str,
    column_filters: list[dict],
    match_col_parents: str,
    match_col_mutants: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter parents DataFrame based on filtering criteria and filter mutants to match.

    This function:
    1. Applies filtering criteria to parents_df
    2. Filters mutants_df to keep only rows matching the filtered parents

    Args:
        parents_df: Original parents DataFrame
        mutants_df: Original mutants DataFrame
        filter_name: Name/identifier for this filter (used in logging)
        column_filters: List of column filter dictionaries. Each dict should have:
            - "column": str - Column name to filter
            - "dtype": Literal["categorical", "float"] - Type of filter
            - For categorical: "values": list[str] - List of values to keep
            - For float: "min": float | None, "max": float | None - Range bounds
        match_col_parents: Column name in parents DataFrame to match on
        match_col_mutants: Column name in mutants DataFrame to match on

    Returns:
        Tuple of (filtered_parents_df, filtered_mutants_df)
    """
    # Apply parent filtering
    filtered_df = parents_df.copy()
    initial_parent_count = len(filtered_df)

    for col_filter in column_filters:
        col_name = col_filter["column"]
        dtype = col_filter["dtype"]

        if col_name not in filtered_df.columns:
            raise ValueError(
                f"Column '{col_name}' not found in parents DataFrame. "
                f"Available columns: {list(filtered_df.columns)}"
            )

        if dtype == "categorical":
            # Categorical filtering
            if "values" not in col_filter:
                raise ValueError(
                    f"Categorical filter for column '{col_name}' must have 'values' key"
                )
            filtered_df = filtered_df[filtered_df[col_name].isin(col_filter["values"])]

        elif dtype == "float":
            # Float range filtering
            mask = pd.Series(True, index=filtered_df.index)

            min_val = col_filter.get("min")
            max_val = col_filter.get("max")

            if min_val is not None:
                mask = mask & (filtered_df[col_name] >= min_val)
            if max_val is not None:
                mask = mask & (filtered_df[col_name] <= max_val)

            filtered_df = filtered_df[mask]
        else:
            raise ValueError(
                f"Unknown filter dtype: {dtype}. Must be 'categorical' or 'float'"
            )

    filtered_parents_df = filtered_df

    # Filter mutants to keep only those matching filtered parents
    parent_ids = set(filtered_parents_df[match_col_parents].unique())
    filtered_mutants_df = mutants_df[
        mutants_df[match_col_mutants].isin(parent_ids)
    ].copy()

    logger.info(
        f"Filter '{filter_name}': "
        f"Parents: {len(filtered_parents_df)}/{initial_parent_count} rows kept, "
        f"Mutants: {len(filtered_mutants_df)}/{len(mutants_df)} rows kept "
        f"(matching {len(parent_ids)} parent IDs)"
    )

    return filtered_parents_df, filtered_mutants_df
