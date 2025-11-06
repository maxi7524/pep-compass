"""Mutation detection and counting functionality."""

import numpy as np
from numba import njit
from typing import Tuple, Union
from collections import Counter
import pandas as pd


@njit
def _find_mutation_positions(
    parent_bytes: np.ndarray,
    mutant_bytes: np.ndarray,
    ngram_size: int,
    allow_overlap: bool,
) -> np.ndarray:
    """Return start indices of differing n-grams."""
    n = len(parent_bytes)
    step = 1 if allow_overlap else ngram_size
    result = []
    for i in range(0, n - ngram_size + 1, step):
        for j in range(ngram_size):
            if parent_bytes[i + j] != mutant_bytes[i + j]:
                result.append(i)
                break
    return np.array(result, dtype=np.int64)


def get_mutation_counts_for_substitution_mutations(
    parent: str,
    mutant: str,
    ngram_size: int,
    allow_ngrams_overlap: bool = True,
    aggregate_over_positions: bool = True,
) -> Union[
    Counter[Tuple[str, str], int],
    Counter[Tuple[str, str, int], int],
]:
    """
    Identify and count substitution mutations between two aligned sequences.

    Args:
        parent, mutant: aligned input sequences (must be same length)
        ngram_size: window size for mutation comparison
        allow_ngrams_overlap: if True, slide window by 1, else by ngram_size
        aggregate_over_positions: if True, aggregate counts across positions

    Returns:
        dict where keys are:
            - (parent_ngram, mutant_ngram): count  if aggregate_over_positions=True
            - (parent_ngram, mutant_ngram, position): count  otherwise
    """
    parent_seq = parent.strip()
    mutant_seq = mutant.strip()

    if len(parent_seq) != len(mutant_seq):
        raise ValueError(
            f"Sequences must have equal length (got {len(parent_seq)} vs {len(mutant_seq)})"
        )

    parent_arr = np.frombuffer(parent_seq.encode(), dtype=np.uint8)
    mutant_arr = np.frombuffer(mutant_seq.encode(), dtype=np.uint8)

    starts = _find_mutation_positions(
        parent_arr, mutant_arr, ngram_size, allow_ngrams_overlap
    )

    # Collect raw (p_ng, m_ng, pos) triplets
    records = [
        (parent_seq[i : i + ngram_size], mutant_seq[i : i + ngram_size], int(i))
        for i in starts
        if parent_seq[i : i + ngram_size] != mutant_seq[i : i + ngram_size]
    ]

    if aggregate_over_positions:
        # Aggregate over positions
        counter = Counter((p, m) for p, m, _ in records)
    else:
        counter = Counter(records)

    return counter


def compute_aggregate_mutation_counter(
    diff_df: pd.DataFrame,
    parent_col: str,
    mutant_col: str,
    ngram_size: int,
    allow_ngrams_overlap: bool = True,
) -> Counter:
    """
    Compute aggregate mutation counter from diff_df (union of all mutation keys).

    Args:
        diff_df: DataFrame with parent and mutant sequence columns
        parent_col: Column name containing parent sequences
        mutant_col: Column name containing mutant sequences
        ngram_size: Size of n-grams for mutation analysis
        allow_ngrams_overlap: Whether to allow overlapping n-grams

    Returns:
        Aggregate Counter with union of all mutation keys
    """
    aggregate_counter = Counter()

    for idx, row in diff_df.iterrows():
        parent = row[parent_col]
        mutant = row[mutant_col]

        # Skip if either is NaN
        if pd.isna(parent) or pd.isna(mutant):
            continue

        # Compute counter for this row
        counter = get_mutation_counts_for_substitution_mutations(
            parent=str(parent),
            mutant=str(mutant),
            ngram_size=ngram_size,
            allow_ngrams_overlap=allow_ngrams_overlap,
            aggregate_over_positions=True,
        )

        # Update aggregate counter (union of all keys with sum of counts)
        aggregate_counter.update(counter)

    return aggregate_counter


def process_bootstrap_mutations(
    bootstrap_df: pd.DataFrame,
    parent_col: str,
    mutant_col: str,
    aggregate_mutation_keys: set,
    ngram_size: int,
    allow_ngrams_overlap: bool = True,
) -> dict[tuple[str, str], Counter]:
    """
    Process bootstrap sample and collect mutation statistics.

    For each row in bootstrap_df, compute mutation counters and append to
    subdict using keys from aggregate_mutation_keys.

    Args:
        bootstrap_df: Bootstrap-sampled DataFrame
        parent_col: Column name containing parent sequences
        mutant_col: Column name containing mutant sequences
        aggregate_mutation_keys: Set of all mutation keys from aggregate counter
        ngram_size: Size of n-grams for mutation analysis
        allow_ngrams_overlap: Whether to allow overlapping n-grams

    Returns:
        Dict mapping (parent, mutant) tuples to Counter objects with
        keys restricted to aggregate_mutation_keys
    """
    bootstrap_counters: dict[tuple[str, str], Counter] = {}

    for idx, row in bootstrap_df.iterrows():
        parent = row[parent_col]
        mutant = row[mutant_col]

        # Skip if either is NaN
        if pd.isna(parent) or pd.isna(mutant):
            continue

        # Compute counter for this row
        counter = get_mutation_counts_for_substitution_mutations(
            parent=str(parent),
            mutant=str(mutant),
            ngram_size=ngram_size,
            allow_ngrams_overlap=allow_ngrams_overlap,
            aggregate_over_positions=True,
        )

        # Filter counter to only include keys from aggregate_mutation_keys
        # and initialize missing keys with 0
        filtered_counter = Counter()
        for key in aggregate_mutation_keys:
            filtered_counter[key] = counter.get(key, 0)

        bootstrap_counters[(parent, mutant)] = filtered_counter

    return bootstrap_counters
