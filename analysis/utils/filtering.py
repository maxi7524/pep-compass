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
