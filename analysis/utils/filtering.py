"""Filtering and deduplication utilities for mutation dataframes."""

import pandas as pd
from typing import Optional
from loguru import logger


def filter_identities(
    df: pd.DataFrame,
    parent_col: str = "parent",
    mutant_col: str = "mutant",
    log_progress: bool = True,
) -> pd.DataFrame:
    """
    Filter out rows where parent sequence equals mutant sequence (identity cases).

    Args:
        df: DataFrame with parent and mutant sequence columns
        parent_col: Column name containing parent sequences
        mutant_col: Column name containing mutant sequences
        log_progress: Whether to log filtering progress

    Returns:
        Filtered DataFrame with identity cases removed
    """
    initial_count = len(df)
    assert mutant_col in df.columns and parent_col in df.columns, KeyError(f"Columns {mutant_col} and {parent_col} not found in DataFrame. Available columns: {list(df.columns)}")
    df_filtered = df[df[mutant_col] != df[parent_col]].copy()
    filtered_count = initial_count - len(df_filtered)

    if log_progress:
        logger.info(
            f"Filtered {filtered_count} identity cases ({parent_col} == {mutant_col}). "
            f"Remaining: {len(df_filtered)}"
        )

    return df_filtered


def filter_length_mismatches(
    df: pd.DataFrame,
    parent_col: str = "parent",
    mutant_col: str = "mutant",
    log_progress: bool = True,
) -> pd.DataFrame:
    """
    Filter out rows where parent and mutant have different lengths after stripping whitespace.

    Args:
        df: DataFrame with parent and mutant sequence columns
        parent_col: Column name containing parent sequences
        mutant_col: Column name containing mutant sequences
        log_progress: Whether to log filtering progress

    Returns:
        Filtered DataFrame with length mismatch cases removed
    """
    initial_count = len(df)
    parent_lengths = df[parent_col].str.strip().str.len()
    mutant_lengths = df[mutant_col].str.strip().str.len()
    df_filtered = df[parent_lengths == mutant_lengths].copy()
    filtered_count = initial_count - len(df_filtered)

    if log_progress:
        logger.info(
            f"Filtered {filtered_count} length mismatch cases (different lengths after strip). "
            f"Remaining: {len(df_filtered)}"
        )

    return df_filtered


def filter_by_length(
    df: pd.DataFrame,
    seq_col: str,
    max_length: Optional[int] = None,
    min_length: Optional[int] = None,
    log_progress: bool = True,
) -> pd.DataFrame:
    """
    Filter sequences by length (keep sequences with length <= max_length and >= min_length).

    Args:
        df: DataFrame with sequence column
        seq_col: Column name containing sequences
        max_length: Maximum sequence length (inclusive). If None, no upper limit.
        min_length: Minimum sequence length (inclusive). If None, no lower limit.
        log_progress: Whether to log filtering progress

    Returns:
        Filtered DataFrame with sequences within the specified length range
    """
    initial_count = len(df)
    seq_lengths = df[seq_col].str.strip().str.len()
    
    mask = pd.Series(True, index=df.index)
    if max_length is not None:
        mask = mask & (seq_lengths <= max_length)
    if min_length is not None:
        mask = mask & (seq_lengths >= min_length)
    
    df_filtered = df[mask].copy()
    filtered_count = initial_count - len(df_filtered)

    if log_progress:
        length_desc = []
        if min_length is not None:
            length_desc.append(f">= {min_length}")
        if max_length is not None:
            length_desc.append(f"<= {max_length}")
        length_str = " and ".join(length_desc) if length_desc else "all lengths"
        logger.info(
            f"Filtered {filtered_count} sequences (length {length_str}). "
            f"Remaining: {len(df_filtered)}"
        )

    return df_filtered


def deduplicate_mutations(
    df: pd.DataFrame,
    parent_col: str = "parent",
    mutant_col: str = "mutant",
    keep: str = "first",
    log_progress: bool = True,
) -> pd.DataFrame:
    """
    Remove duplicate rows where both parent and mutant sequences are the same.

    Args:
        df: DataFrame with parent and mutant sequence columns
        parent_col: Column name containing parent sequences
        mutant_col: Column name containing mutant sequences
        keep: Which duplicate to keep ('first', 'last', or False to drop all)
        log_progress: Whether to log deduplication progress

    Returns:
        DataFrame with duplicate rows removed
    """
    initial_count = len(df)
    df_deduped = df.drop_duplicates(subset=[parent_col, mutant_col], keep=keep).copy()
    dedup_count = initial_count - len(df_deduped)

    if log_progress:
        logger.info(
            f"Removed {dedup_count} duplicate rows (same {parent_col} and {mutant_col}). "
            f"Remaining: {len(df_deduped)}"
        )

    return df_deduped


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
