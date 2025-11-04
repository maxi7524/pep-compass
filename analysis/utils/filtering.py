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
