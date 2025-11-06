"""Statistical analysis functionality for bootstrap samples."""

import pandas as pd
import numpy as np
from typing import Optional, Generator
from collections import Counter
from scipy.stats import rankdata

from .bootstrap import bootstrap_df_generator
from .mutations import process_bootstrap_mutations


def compute_mutation_statistics_from_df(
    diff_df: pd.DataFrame,
    aggregate_counter: Counter,
    parent_col: str,
    mutant_col: str,
    value_cols: list[str],
    ngram_size: int,
    allow_ngrams_overlap: bool = True,
) -> dict:
    """
    Compute mutation statistics dictionary from a single DataFrame.

    This function processes a DataFrame and returns a dict mapping mutation keys
    to dictionaries with column names as keys and lists of values as values.

    Args:
        diff_df: DataFrame to process (typically a single bootstrap sample)
        aggregate_counter: Aggregate Counter with all mutation keys
        parent_col: Column name containing parent sequences
        mutant_col: Column name containing mutant sequences
        value_cols: List of column names to collect statistics for
        ngram_size: Size of n-grams for mutation analysis
        allow_ngrams_overlap: Whether to allow overlapping n-grams

    Returns:
        Dict mapping mutation keys (tuples) to dicts with column names as keys
        and lists of values as values
    """
    aggregate_keys = set(aggregate_counter.keys())

    # Process mutations
    bootstrap_counters = process_bootstrap_mutations(
        bootstrap_df=diff_df,
        parent_col=parent_col,
        mutant_col=mutant_col,
        aggregate_mutation_keys=aggregate_keys,
        ngram_size=ngram_size,
        allow_ngrams_overlap=allow_ngrams_overlap,
    )

    # Initialize result dict: mutation_key -> {col_name: [values]}
    result: dict[tuple[str, str], dict[str, list]] = {}
    for key in aggregate_keys:
        result[key] = {col: [] for col in value_cols}

    # Collect values for each row in DataFrame
    for idx, row in diff_df.iterrows():
        parent = row[parent_col]
        mutant = row[mutant_col]

        # Skip if either is NaN
        if pd.isna(parent) or pd.isna(mutant):
            continue

        pair_key = (parent, mutant)

        # Get counter for this pair
        counter = bootstrap_counters.get(pair_key, Counter())

        # For each mutation in this row, add the column values
        for mutation_key, count in counter.items():
            if count > 0:  # Only if mutation is present
                for col in value_cols:
                    # Append the value 'count' times (once per occurrence)
                    result[mutation_key][col].extend([row[col]] * count)

    return result


def compute_ranks_for_single_sample(
    stats_dict: dict,
    aggregate_cols: list[str],
    aggregate_counter: Counter,
    aggregation_func: str = "mean",
) -> dict[tuple[str, str], dict[str, float]]:
    """
    Compute ranks for a single mutation statistics dictionary.

    This function takes a single stats dictionary (e.g., from one bootstrap sample)
    and computes ranks for each mutation.

    Args:
        stats_dict: Dictionary mapping mutation keys to dicts with column value lists
        aggregate_cols: List of column names to aggregate and rank
        aggregate_counter: Counter with all mutation keys to include in output
        aggregation_func: Function to aggregate values within a sample ('mean', 'sum', etc.)

    Returns:
        Dict mapping mutation keys to dicts with rank values per column.
        Mutations that didn't appear in the sample will have NaN ranks.
    """
    all_mutation_keys = list(aggregate_counter.keys())

    # Initialize aggregation function
    if aggregation_func == "mean":
        agg_fn = lambda x: np.mean(x) if len(x) > 0 else np.nan
    elif aggregation_func == "sum":
        agg_fn = lambda x: np.sum(x) if len(x) > 0 else np.nan
    else:
        raise ValueError(f"Unknown aggregation function: {aggregation_func}")

    # Aggregate values per mutation
    sample_aggregated = {}
    for mutation_key in all_mutation_keys:
        sample_aggregated[mutation_key] = {}
        for col in aggregate_cols:
            # Get all values for this mutation in this sample
            values = []
            if mutation_key in stats_dict and col in stats_dict[mutation_key]:
                values = stats_dict[mutation_key][col]
            # Aggregate values for this mutation
            sample_aggregated[mutation_key][col] = agg_fn(values)

    # Compute ranks per column using scipy
    rank_dict = {
        k: {f"{col}_rank": np.nan for col in aggregate_cols} for k in all_mutation_keys
    }

    for col in aggregate_cols:
        # Extract aggregated values for all mutations
        values = np.array(
            [
                (
                    sample_aggregated[k][col]
                    if not np.isnan(sample_aggregated[k][col])
                    else np.nan
                )
                for k in all_mutation_keys
            ]
        )

        # Compute ranks using scipy
        ranks = rankdata(values, method="average", nan_policy="omit")

        # Store ranks for mutations that appeared in this sample
        valid_mask = ~np.isnan(values)
        for idx, mutation_key in enumerate(all_mutation_keys):
            if valid_mask[idx]:
                rank_dict[mutation_key][f"{col}_rank"] = float(ranks[idx])

    return rank_dict


def mutation_statistics_generator(
    diff_df: pd.DataFrame,
    aggregate_counter: Counter,
    parent_col: str,
    mutant_col: str,
    value_cols: list[str],
    ngram_size: int,
    allow_ngrams_overlap: bool = True,
    n_bootstrap_samples: Optional[int] = None,
    bootstrap_group_cols: Optional[list[str]] = None,
    stratify_cols: Optional[list[str]] = None,
    bootstrap_frac: float = 1.0,
    seed: Optional[int] = None,
) -> Generator[dict, None, None]:
    """
    Generator that yields mutation-based statistics for bootstrap samples.

    For each bootstrap sample:
        1. Processes mutations using aggregate counter keys
        2. Groups values by mutation key
        3. Yields dict mapping mutation keys to column value lists

    Args:
        diff_df: Original diff DataFrame
        aggregate_counter: Aggregate Counter with all mutation keys
        parent_col: Column name containing parent sequences
        mutant_col: Column name containing mutant sequences
        value_cols: List of column names to collect statistics for
        ngram_size: Size of n-grams for mutation analysis
        allow_ngrams_overlap: Whether to allow overlapping n-grams
        n_bootstrap_samples: Number of bootstrap samples (None for infinite)
        bootstrap_group_cols: Columns to group by for bootstrapping
        stratify_cols: Columns to stratify by for bootstrapping
        bootstrap_frac: Fraction of data to use in bootstrap samples
        seed: Random seed for bootstrapping

    Yields:
        Dict mapping mutation keys (tuples) to dicts with column names as keys
        and lists of values as values
    """
    aggregate_keys = set(aggregate_counter.keys())

    # Create bootstrap generator
    if n_bootstrap_samples is None:
        # Create infinite generator by using a large number and breaking manually
        n_bootstrap_samples = 10**9
    else:
        n_bootstrap_samples = n_bootstrap_samples

    bootstrap_gen = bootstrap_df_generator(
        df=diff_df,
        n_bootstrap_samples=n_bootstrap_samples,
        bootstrap_group_cols=bootstrap_group_cols,
        stratify_cols=stratify_cols,
        bootstrap_frac=bootstrap_frac,
        seed=seed,
    )

    for bootstrap_df in bootstrap_gen:
        yield compute_mutation_statistics_from_df(
            diff_df=bootstrap_df,
            aggregate_counter=aggregate_counter,
            parent_col=parent_col,
            mutant_col=mutant_col,
            value_cols=value_cols,
            ngram_size=ngram_size,
            allow_ngrams_overlap=allow_ngrams_overlap,
        )


def compute_bootstrap_sample_ranks(
    stats_generator: Generator[dict, None, None],
    aggregate_cols: list[str],
    aggregate_counter: Counter,
    aggregation_func: str = "mean",
) -> dict[tuple[str, str], dict[str, list]]:
    """
    Compute ranks for each bootstrap sample and accumulate in lists.

    For each bootstrap sample:
    1. Compute ranks using compute_ranks_for_single_sample
    2. Append ranks to accumulated lists

    All lists have the same length (number of bootstrap samples).
    Lists contain NaN where mutations didn't appear, and rank values where they did.

    Args:
        stats_generator: Generator yielding dicts with mutation keys and column values
        aggregate_cols: List of column names to aggregate and rank
        aggregate_counter: Counter with all mutation keys to include in output
        aggregation_func: Function to aggregate values within a sample ('mean', 'sum', etc.)

    Returns:
        Dict mapping mutation keys to dicts with lists of rank values per column.
        All lists have the same length (number of bootstrap samples processed).
        Contains NaN for bootstrap samples where mutation didn't appear.
    """
    all_mutation_keys = list(aggregate_counter.keys())

    # Initialize lists for each mutation/column
    rank_lists = {
        k: {f"{col}_rank": [] for col in aggregate_cols} for k in all_mutation_keys
    }

    # Process each bootstrap sample
    for stats in stats_generator:
        # Compute ranks for this single sample
        sample_ranks = compute_ranks_for_single_sample(
            stats_dict=stats,
            aggregate_cols=aggregate_cols,
            aggregate_counter=aggregate_counter,
            aggregation_func=aggregation_func,
        )

        # Append ranks to the accumulated lists
        for mutation_key in all_mutation_keys:
            for col in aggregate_cols:
                rank_lists[mutation_key][f"{col}_rank"].append(
                    sample_ranks[mutation_key][f"{col}_rank"]
                )

    return rank_lists
