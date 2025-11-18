"""Bootstrap sampling functionality."""

import pandas as pd
import numpy as np
from sklearn.utils import resample
from typing import Optional, Generator
from collections import Counter


def bootstrap_df_generator(
    df: pd.DataFrame,
    n_bootstrap_samples: int,
    bootstrap_group_cols: Optional[list[str]] = None,
    stratify_cols: Optional[list[str]] = None,
    bootstrap_frac: float = 1.0,
    seed: Optional[int] = None,
) -> Generator[pd.DataFrame, None, None]:
    """Yield bootstrap-sampled DataFrames (grouped or rowwise, optional stratification)."""
    rng = np.random.default_rng(seed)
    n_rows = len(df)
    ilocs = np.arange(n_rows)

    # --- determine grouping ---
    if bootstrap_group_cols:
        group_keys = df[bootstrap_group_cols].astype(str).agg("_".join, axis=1)
    else:
        group_keys = pd.Series(ilocs.astype(str), index=df.index)

    # --- construct group table ---
    group_table = (
        pd.DataFrame({"_group_key": group_keys})
        .value_counts()
        .reset_index(name="weight")
    )

    # --- optional stratification ---
    stratify = None
    if stratify_cols:
        stratify = (
            df.groupby(group_keys)[stratify_cols]
            .first()
            .astype(str)
            .agg("_".join, axis=1)
            .reindex(group_table["_group_key"])
            .values
        )

    # --- precompute mapping group → iloc indices ---
    group_to_ilocs = (
        pd.DataFrame({"_group_key": group_keys, "_iloc": ilocs})
        .groupby("_group_key")["_iloc"]
        .apply(np.array)
        .to_dict()
    )

    # --- sampling parameters ---
    n_groups = len(group_table)
    n_samples = int(n_groups * bootstrap_frac)
    weights = group_table["weight"] if bootstrap_group_cols else None

    # --- sampling loop ---
    for _ in range(n_bootstrap_samples):
        rs = int(rng.integers(0, 1e9))
        sampled_groups = resample(
            group_table,
            replace=True,
            n_samples=n_samples,
            stratify=stratify,
            sample_weight=weights,
            random_state=rs,
        )

        iloc_indices = np.concatenate(
            [group_to_ilocs[gk] for gk in sampled_groups["_group_key"]]
        )
        yield df.iloc[iloc_indices].copy()

from collections import defaultdict
from typing import List, Dict, Any


def merge_bootstrap_samples(bootstrap_samples: List[Dict[Any, Dict[str, float]]]) -> Dict[Any, Dict[str, List[float]]]:
    """
    Merge multiple bootstrap samples into a single dictionary.
    
    Each bootstrap sample is a dictionary mapping mutation keys to dictionaries
    of bacteria ranks. This function collects all rank values for each mutation
    key and bacteria into lists.
    
    Args:
        bootstrap_samples: List of bootstrap sample dictionaries, where each
            sample maps mutation keys to dictionaries of bacteria rank values.
    
    Returns:
        Dictionary where each mutation key maps to a dictionary of bacteria
        names mapping to lists of rank values from all bootstrap samples.
    
    Example:
        Input:
        [
            {('L', 'N'): {'A. baumannii_diff_rank': 304.0, ...}},
            {('L', 'N'): {'A. baumannii_diff_rank': 310.0, ...}},
        ]
        
        Output:
        {
            ('L', 'N'): {
                'A. baumannii_diff_rank': [304.0, 310.0, ...],
                ...
            }
        }
    """
    merged = defaultdict(lambda: defaultdict(list))
    
    for sample in bootstrap_samples:
        for mutation_key, ranks_dict in sample.items():
            for bacteria, rank_value in ranks_dict.items():
                merged[mutation_key][bacteria].append(rank_value)
    
    # Convert defaultdict to regular dict
    return {k: dict(v) for k, v in merged.items()}