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
