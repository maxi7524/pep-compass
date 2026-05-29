"""Utility functions for data processing."""

import pandas as pd
import numpy as np
from typing import Literal, Optional
from pathlib import Path
from loguru import logger


def compute_diff(
    merged_df: pd.DataFrame | None = None,
    parents_df: pd.DataFrame | None = None,
    mutants_df: pd.DataFrame | None = None,
    match_col_parents: str | None = None,
    match_col_mutants: str | None = None,
    value_cols: list[str] | None = None,
    value_preprocessing: Optional[Literal["log2"]] = None,
    add_relative: bool = False,
) -> pd.DataFrame:
    """
    Compute the difference between the mutants and parents for each value column.

    Can work with either:
    - A pre-merged dataframe (merged_df) with _mutants and _parents suffixes
    - Separate parent and mutant dataframes (will merge them)

    Args:
        merged_df: Pre-merged DataFrame with _mutants and _parents suffixes (optional)
        parents_df: Parents DataFrame (required if merged_df is None)
        mutants_df: Mutants DataFrame (required if merged_df is None)
        match_col_parents: Column name to match parents (required if merging)
        match_col_mutants: Column name to match mutants (required if merging)
        value_cols: List of column names to compute differences for
        value_preprocessing: Preprocessing for values: None or "log2"
        add_relative: Whether to compute relative differences

    Returns:
        DataFrame with diff columns added
    """
    if merged_df is not None:
        # Use pre-merged dataframe
        if parents_df is not None or mutants_df is not None:
            raise ValueError("Cannot provide both merged_df and separate dataframes")
        if value_cols is None:
            raise ValueError("value_cols must be provided when using merged_df")
        merged = merged_df.copy()
    else:
        # Merge separate dataframes
        if parents_df is None or mutants_df is None:
            raise ValueError("Must provide either merged_df or both parents_df and mutants_df")
        if match_col_parents is None or match_col_mutants is None:
            raise ValueError("match_col_parents and match_col_mutants required when merging")
        if value_cols is None:
            raise ValueError("value_cols must be provided")
        merged = mutants_df.merge(
            parents_df,
            left_on=match_col_mutants,
            right_on=match_col_parents,
            suffixes=("_mutants", "_parents"),
        )

    if value_preprocessing is None:
        vprep_fn = lambda x: x
    elif value_preprocessing == "log2":
        vprep_fn = np.log2
    else:
        raise ValueError(f"Invalid value_preprocessing: {value_preprocessing}")
    if value_preprocessing is None:
        value_preprocessing = ""
    else:
        value_preprocessing = "_" + value_preprocessing

    for value_col in value_cols:
        mutant_col = f"{value_col}_mutants"
        parent_col = f"{value_col}_parents"
        
        if mutant_col not in merged.columns:
            raise ValueError(f"Column '{mutant_col}' not found in merged DataFrame")
        if parent_col not in merged.columns:
            raise ValueError(f"Column '{parent_col}' not found in merged DataFrame")
        
        merged[f"{value_col}{value_preprocessing}_diff"] = vprep_fn(
            merged[mutant_col]
        ) - vprep_fn(merged[parent_col])
        if add_relative:
            merged[f"{value_col}{value_preprocessing}_diff_relative"] = merged[
                f"{value_col}{value_preprocessing}_diff"
            ] / vprep_fn(merged[parent_col])
    return merged


def compute_total_nonidentity_ngram_mutations(alphabet: str, ngram_size: int) -> int:
    """Compute total possible non-identity ngram mutations: |alphabet|^ngram_size * (|alphabet|^ngram_size - 1)."""
    total_ngrams = len(alphabet) ** ngram_size
    return total_ngrams * (total_ngrams - 1)
