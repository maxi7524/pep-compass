#!/usr/bin/env python3
"""Compute bootstrap statistics for mutation analysis."""
import sys
from pathlib import Path
import pandas as pd
from loguru import logger
import pickle

from .config import (
    CombinedPairConfig,
    CombinedGroupConfig,
    FilteringConfig,
    AnalysesConfig,
    DiffAnalysisConfig,
    TransitionCountsAnalysisConfig,
    MutationFilterConfig,
    MutationConfig,
    RankConfig,
    DiffConfig,
    load_config,
)
from .utils import compute_diff
from .mutations import compute_aggregate_mutation_counter
from .analysis import (
    compute_ranks_for_single_sample,
    compute_mutation_statistics_from_df,
)
from .bootstrap import bootstrap_df_generator
from utils.filtering import (
    filter_identities,
    filter_length_mismatches,
    deduplicate_mutations,
)
from collections import Counter
from typing import Literal
from tqdm import tqdm


def combine_pairs_into_datasets(
    pairs: list[CombinedPairConfig],
    match_col_parents: str,
    match_col_mutants: str,
) -> pd.DataFrame:
    """
    Combine multiple parent-mutant pairs into a single merged dataset.

    Loads all parent and mutant files from pairs list, applies column remapping if provided,
    matches each pair using global match columns, and concatenates all matched pairs into
    a single merged dataframe.

    Args:
        pairs: List of pair configurations to combine
        match_col_parents: Global column name to match parents (used after remapping)
        match_col_mutants: Global column name to match mutants (used after remapping)

    Returns:
        Merged DataFrame with columns having _mutants and _parents suffixes
    """
    all_matched_dfs = []

    for pair in pairs:
        parents_path = pair.parent
        mutants_path = pair.mutant

        if not parents_path.exists():
            raise FileNotFoundError(f"Parents dataset file not found at {parents_path}")

        if not mutants_path.exists():
            raise FileNotFoundError(f"Mutants dataset file not found at {mutants_path}")

        logger.info(f"Loading parents from: {parents_path}")
        parents_df = pd.read_csv(parents_path, sep=pair.parent_csv_separator)
        
        # Apply column remapping if provided
        if pair.parent_column_rename_map:
            parents_df = parents_df.rename(columns=pair.parent_column_rename_map)

        logger.info(f"Loading mutants from: {mutants_path}")
        mutants_df = pd.read_csv(mutants_path)
        
        # Apply column remapping if provided
        if pair.mutant_column_rename_map:
            mutants_df = mutants_df.rename(columns=pair.mutant_column_rename_map)

        # Match pairs using the global match columns (after remapping, all pairs use same column names)
        merged = mutants_df.merge(
            parents_df,
            left_on=match_col_mutants,
            right_on=match_col_parents,
            suffixes=("_mutants", "_parents"),
        )

        # Release individual dataframes from memory
        del parents_df, mutants_df

        # Deduplicate this pair's result based on the global match columns (keep first occurrence)
        # This ensures we don't have duplicate parent-mutant pairs within this pair
        merged = merged.drop_duplicates(
            subset=[match_col_parents, match_col_mutants], keep="first"
        )

        all_matched_dfs.append(merged)

    if not all_matched_dfs:
        raise ValueError("No valid pairs found to combine")

    # Concatenate all matched DataFrames
    combined_df = pd.concat(all_matched_dfs, ignore_index=True)
    
    # Release list of individual matched dataframes from memory
    del all_matched_dfs

    logger.info(
        f"Combined dataset: {len(combined_df)} rows after merging and deduplication"
    )

    return combined_df


def filter_merged_parents(
    merged_df: pd.DataFrame,
    filter_name: str,
    column_filters: list[dict],
) -> pd.DataFrame:
    """
    Filter merged dataframe based on parent column filtering criteria.

    Filters rows where parent columns (with _parents suffix) match the criteria.

    Args:
        merged_df: Merged DataFrame with _mutants and _parents suffixes
        filter_name: Name/identifier for this filter (used in logging)
        column_filters: List of column filter dictionaries. Each dict should have:
            - "column": str - Column name to filter (will look for column_parents)
            - "dtype": Literal["categorical", "float"] - Type of filter
            - For categorical: "values": list[str] - List of values to keep
            - For float: "min": float | None, "max": float | None - Range bounds

    Returns:
        Filtered merged DataFrame
    """
    filtered_df = merged_df.copy()
    initial_count = len(filtered_df)

    for col_filter in column_filters:
        col_name = col_filter["column"]
        dtype = col_filter["dtype"]
        
        # Look for column with _parents suffix
        parent_col_name = f"{col_name}_parents"
        if parent_col_name not in filtered_df.columns:
            raise ValueError(
                f"Column '{parent_col_name}' not found in merged DataFrame. "
                f"Available columns: {list(filtered_df.columns)}"
            )

        if dtype == "categorical":
            if "values" not in col_filter:
                raise ValueError(
                    f"Categorical filter for column '{col_name}' must have 'values' key"
                )
            filtered_df = filtered_df[filtered_df[parent_col_name].isin(col_filter["values"])]

        elif dtype == "float":
            mask = pd.Series(True, index=filtered_df.index)
            min_val = col_filter.get("min")
            max_val = col_filter.get("max")

            if min_val is not None:
                mask = mask & (filtered_df[parent_col_name] >= min_val)
            if max_val is not None:
                mask = mask & (filtered_df[parent_col_name] <= max_val)

            filtered_df = filtered_df[mask]
        else:
            raise ValueError(
                f"Unknown filter dtype: {dtype}. Must be 'categorical' or 'float'"
            )

    logger.info(
        f"Filter '{filter_name}': "
        f"{len(filtered_df)}/{initial_count} rows kept after parent filtering"
    )

    return filtered_df


def apply_post_combination_filters(
    diff_df: pd.DataFrame,
    filter_config: MutationFilterConfig,
    mutation_config: MutationConfig,
) -> pd.DataFrame:
    """
    Apply post-combination filters to diff DataFrame.

    Args:
        diff_df: Diff DataFrame to filter
        filter_config: Mutation filter configuration
        mutation_config: Mutation configuration for column names

    Returns:
        Filtered diff DataFrame
    """
    filtered_df = diff_df.copy()
    initial_count = len(filtered_df)

    if filter_config.filter_identities:
        filtered_df = filter_identities(
            filtered_df,
            parent_col=mutation_config.parent_sequence_col,
            mutant_col=mutation_config.mutant_sequence_col,
            log_progress=True,
        )

    if filter_config.filter_length_mismatch:
        filtered_df = filter_length_mismatches(
            filtered_df,
            parent_col=mutation_config.parent_sequence_col,
            mutant_col=mutation_config.mutant_sequence_col,
            log_progress=True,
        )

    if filter_config.deduplicate:
        filtered_df = deduplicate_mutations(
            filtered_df,
            parent_col=mutation_config.parent_sequence_col,
            mutant_col=mutation_config.mutant_sequence_col,
            keep="first",
            log_progress=True,
        )

    logger.info(
        f"After post-combination filtering: {len(filtered_df)} rows "
        f"(removed {initial_count - len(filtered_df)})"
    )

    return filtered_df


def run_diff_analysis(
    diff_df: pd.DataFrame,
    analysis_config: DiffAnalysisConfig,
    output_dir: Path,
    dataset_name: str,
    mutation_config: MutationConfig | None = None,
    diff_config: DiffConfig | None = None,
    global_ngram_size: int | None = None,
) -> None:
    """
    Run diff analysis with configurable outputs.

    Args:
        diff_df: Diff DataFrame
        analysis_config: Diff analysis configuration
        output_dir: Output directory
        dataset_name: Name for output files
        mutation_config: Mutation configuration (required for ranks computation)
        diff_config: Diff configuration (required for ranks computation)
    """
    if not analysis_config.enabled:
        logger.info("Diff analysis is disabled, skipping")
        return

    # Save diff values if requested
    if analysis_config.output.values:
        diff_output_path = output_dir / "diff_values.csv"
        diff_df.to_csv(diff_output_path, index=False)
        logger.success(f"Saved diff values to: {diff_output_path}")

    # Compute and save ranks if requested
    if analysis_config.output.ranks:
        if analysis_config.output.ranks_config is None:
            raise ValueError("ranks_config must be provided if ranks is True")
        if mutation_config is None:
            raise ValueError(
                "mutation_config must be provided for diff ranks computation. "
                "This requires mutation configuration to aggregate diff values by mutations."
            )
        if diff_config is None:
            raise ValueError(
                "diff_config must be provided for diff ranks computation. "
                "This is needed to determine which diff value columns to rank."
            )

        logger.info("Computing ranks from diff values...")

        # Determine diff value columns to rank
        value_preprocessing_suffix = (
            f"_{diff_config.value_preprocessing}"
            if diff_config.value_preprocessing
            else ""
        )
        diff_value_cols = [
            f"{col}{value_preprocessing_suffix}_diff"
            for col in diff_config.value_cols
        ]
        if diff_config.add_relative:
            diff_value_cols.extend([
                f"{col}{value_preprocessing_suffix}_diff_relative"
                for col in diff_config.value_cols
            ])

        # Verify all diff value columns exist in diff_df
        missing_cols = [col for col in diff_value_cols if col not in diff_df.columns]
        if missing_cols:
            raise ValueError(
                f"Diff value columns not found in diff_df: {missing_cols}. "
                f"Available columns: {list(diff_df.columns)}"
            )

        # Get ngram_size from mutation_config or global
        ngram_size = mutation_config.ngram_size if mutation_config.ngram_size is not None else global_ngram_size
        if ngram_size is None:
            raise ValueError("ngram_size must be provided either in mutation_config or as global_ngram_size")

        # Compute aggregate mutation counter
        logger.info("Computing aggregate mutation counter for diff ranks...")
        aggregate_counter = compute_aggregate_mutation_counter(
            diff_df=diff_df,
            parent_col=mutation_config.parent_sequence_col,
            mutant_col=mutation_config.mutant_sequence_col,
            ngram_size=ngram_size,
            allow_ngrams_overlap=mutation_config.allow_ngrams_overlap,
        )

        # Compute statistics (aggregate diff values by mutations)
        logger.info("Computing mutation statistics from diff values...")
        stats = compute_mutation_statistics_from_df(
            diff_df=diff_df,
            aggregate_counter=aggregate_counter,
            parent_col=mutation_config.parent_sequence_col,
            mutant_col=mutation_config.mutant_sequence_col,
            value_cols=diff_value_cols,
            ngram_size=ngram_size,
            allow_ngrams_overlap=mutation_config.allow_ngrams_overlap,
        )

        # Compute ranks from statistics
        logger.info("Computing ranks from mutation statistics...")
        ranks = compute_ranks_for_single_sample(
            stats_dict=stats,
            aggregate_cols=diff_value_cols,
            aggregate_counter=aggregate_counter,
            aggregation_func=analysis_config.output.ranks_config.aggregation_func,
        )

        # Save ranks
        ranks_output_path = output_dir / "diff_ranks.pkl"
        with open(ranks_output_path, "wb") as f:
            pickle.dump(ranks, f)
        logger.success(f"Saved diff ranks to: {ranks_output_path}")


def run_transition_counts_analysis(
    diff_df: pd.DataFrame,
    analysis_config: TransitionCountsAnalysisConfig,
    output_dir: Path,
    dataset_name: str,
    diff_config: DiffConfig | None = None,
    global_ngram_size: int | None = None,
) -> Counter | None:
    """
    Run transition counts analysis with configurable outputs.

    Args:
        diff_df: Diff DataFrame
        analysis_config: Transition counts analysis configuration
        output_dir: Output directory
        dataset_name: Name for output files
        diff_config: Diff configuration (required if statistics or ranks are enabled)

    Returns:
        Aggregate mutation counter, or None if analysis is disabled
    """
    if not analysis_config.enabled:
        logger.info("Transition counts analysis is disabled, skipping")
        return None

    # Get ngram_size from config or global
    ngram_size = analysis_config.config.ngram_size if analysis_config.config.ngram_size is not None else global_ngram_size
    if ngram_size is None:
        raise ValueError("ngram_size must be provided either in transition_counts.config or as global_ngram_size")

    # Compute aggregate mutation counter
    logger.info("Computing aggregate mutation counter...")
    aggregate_counter = compute_aggregate_mutation_counter(
        diff_df=diff_df,
        parent_col=analysis_config.config.parent_sequence_col,
        mutant_col=analysis_config.config.mutant_sequence_col,
        ngram_size=ngram_size,
        allow_ngrams_overlap=analysis_config.config.allow_ngrams_overlap,
    )
    logger.info(f"Total unique mutations in aggregate: {len(aggregate_counter)}")

    # Save counts if requested
    if analysis_config.output.counts:
        counts_output_path = output_dir / "transition_counts.pkl"
        with open(counts_output_path, "wb") as f:
            pickle.dump(aggregate_counter, f)
        logger.success(f"Saved transition counts to: {counts_output_path}")

    # Determine if we need to compute statistics (for statistics output or ranks)
    need_statistics = analysis_config.output.statistics or analysis_config.output.ranks
    stats = None
    diff_value_cols = None

    if need_statistics:
        if diff_config is None:
            raise ValueError(
                "diff_config is required for transition_counts statistics/ranks computation. "
                "Diff analysis must be enabled to compute statistics or ranks."
            )

        # Determine value columns to use (those with _diff suffix)
        value_preprocessing_suffix = (
            f"_{diff_config.value_preprocessing}"
            if diff_config.value_preprocessing
            else ""
        )
        diff_value_cols = [
            f"{col}{value_preprocessing_suffix}_diff"
            for col in diff_config.value_cols
        ]

        # Verify all diff value columns exist in diff_df
        missing_cols = [col for col in diff_value_cols if col not in diff_df.columns]
        if missing_cols:
            raise ValueError(
                f"Diff value columns not found in diff_df: {missing_cols}. "
                f"Available columns: {list(diff_df.columns)}"
            )

        # Compute statistics per mutation
        logger.info("Computing transition statistics per mutation...")
        stats = compute_mutation_statistics_from_df(
                diff_df=diff_df,
                aggregate_counter=aggregate_counter,
                parent_col=analysis_config.config.parent_sequence_col,
                mutant_col=analysis_config.config.mutant_sequence_col,
                value_cols=diff_value_cols,
                ngram_size=ngram_size,
                allow_ngrams_overlap=analysis_config.config.allow_ngrams_overlap,
            )

        # Save statistics if requested
        if analysis_config.output.statistics:
            stats_output_path = output_dir / "transition_counts_statistics.pkl"
            with open(stats_output_path, "wb") as f:
                pickle.dump(stats, f)
            logger.success(f"Saved transition statistics to: {stats_output_path}")

    # Compute and save ranks if requested
    if analysis_config.output.ranks:
        if analysis_config.output.ranks_config is None:
            raise ValueError("ranks_config must be provided if ranks is True")
        if stats is None or diff_value_cols is None:
            raise ValueError(
                "Statistics and diff_value_cols must be computed for ranks. "
                "This should not happen if diff_config is provided."
            )

        logger.info("Computing ranks from transition statistics...")

        # Compute ranks from statistics
        ranks = compute_ranks_for_single_sample(
            stats_dict=stats,
            aggregate_cols=diff_value_cols,
            aggregate_counter=aggregate_counter,
            aggregation_func=analysis_config.output.ranks_config.aggregation_func,
        )

        # Save ranks
        ranks_output_path = output_dir / "transition_counts_ranks.pkl"
        with open(ranks_output_path, "wb") as f:
            pickle.dump(ranks, f)
        logger.success(f"Saved transition ranks to: {ranks_output_path}")

    return aggregate_counter


def run_bootstrap_analysis_for_type(
    bootstrap_dfs: list[pd.DataFrame],
    diff_df: pd.DataFrame,
    aggregate_counter: Counter,
    analysis_type: Literal["diff", "transition_counts"],
    mutation_config: MutationConfig,
    rank_config: RankConfig,
    diff_config: DiffConfig,
    output_dir: Path,
    seed: int,
    global_ngram_size: int | None = None,
) -> None:
    """
    Run bootstrap analysis for a single analysis type using pre-generated bootstrap datasets.

    Args:
        bootstrap_dfs: List of bootstrap DataFrames (pre-generated)
        diff_df: Original diff DataFrame
        aggregate_counter: Aggregate mutation counter
        analysis_type: Type of analysis ("diff" or "transition_counts")
        mutation_config: Mutation configuration
        rank_config: Rank configuration
        diff_config: Diff configuration
        output_dir: Output directory
        seed: Random seed (used in directory name)
    """
    # Determine value columns to use (those with _diff suffix)
    value_preprocessing_suffix = (
        f"_{diff_config.value_preprocessing}"
        if diff_config.value_preprocessing
        else ""
    )
    diff_value_cols = [
        f"{col}{value_preprocessing_suffix}_diff"
        for col in diff_config.value_cols
    ]

    # Verify all diff value columns exist in diff_df
    missing_cols = [col for col in diff_value_cols if col not in diff_df.columns]
    if missing_cols:
        raise ValueError(
            f"Diff value columns not found in diff_df: {missing_cols}. "
            f"Available columns: {list(diff_df.columns)}"
        )

    # Create output subdirectory for this analysis type
    bootstrap_output_dir = output_dir / "bootstrap" / analysis_type
    bootstrap_output_dir.mkdir(parents=True, exist_ok=True)


    # Process bootstrap samples with progress bar
    n_samples = len(bootstrap_dfs)
    
    # Create subdirectory for bootstrap samples with seed and number of samples in name
    bootstrap_samples_dir = bootstrap_output_dir / f"seed{seed}_n{n_samples}"
    bootstrap_samples_dir.mkdir(parents=True, exist_ok=True)
    with tqdm(
        total=n_samples,
        desc=f"Bootstrap samples ({analysis_type})",
        unit="sample",
        ncols=100,
        leave=True,
        position=0,
    ) as pbar:
        for sample_idx, bootstrap_df in enumerate(bootstrap_dfs, 1):
            # Get ngram_size from mutation_config or global
            ngram_size = mutation_config.ngram_size if mutation_config.ngram_size is not None else global_ngram_size
            if ngram_size is None:
                raise ValueError("ngram_size must be provided either in mutation_config or as global_ngram_size")

            # Compute statistics for this bootstrap sample
            stats = compute_mutation_statistics_from_df(
                diff_df=bootstrap_df,
                aggregate_counter=aggregate_counter,
                parent_col=mutation_config.parent_sequence_col,
                mutant_col=mutation_config.mutant_sequence_col,
                value_cols=diff_value_cols,
                ngram_size=ngram_size,
                allow_ngrams_overlap=mutation_config.allow_ngrams_overlap,
            )

            # Compute ranks for this single sample
            sample_ranks = compute_ranks_for_single_sample(
                stats_dict=stats,
                aggregate_cols=diff_value_cols,
                aggregate_counter=aggregate_counter,
                aggregation_func=rank_config.aggregation_func,
            )

            # Save this bootstrap sample's ranks
            sample_file = bootstrap_samples_dir / f"{sample_idx}.pkl"
            with open(sample_file, "wb") as f:
                pickle.dump(sample_ranks, f)

            pbar.update(1)
            pbar.refresh()

    logger.success(
        f"Saved {n_samples} bootstrap sample files to {bootstrap_samples_dir}"
    )


def check_outputs_exist(
    output_dir: Path,
    analyses: AnalysesConfig,
    bootstrap_enabled: bool = False,
    bootstrap_seed: int | None = None,
    n_bootstrap_samples: int | None = None,
) -> bool:
    """
    Check if all expected output files already exist.

    Args:
        output_dir: Output directory to check
        analyses: Analyses configuration
        bootstrap_enabled: Whether bootstrap is enabled
        bootstrap_seed: Bootstrap seed (for checking bootstrap outputs)
        n_bootstrap_samples: Number of bootstrap samples (for checking bootstrap outputs)

    Returns:
        True if all expected outputs exist, False otherwise
    """
    # Check diff outputs
    if analyses.diff.enabled:
        if analyses.diff.output.values:
            if not (output_dir / "diff_values.csv").exists():
                return False
        if analyses.diff.output.ranks:
            if not (output_dir / "diff_ranks.pkl").exists():
                return False

    # Check transition_counts outputs
    if analyses.transition_counts.enabled:
        if analyses.transition_counts.output.counts:
            if not (output_dir / "transition_counts.pkl").exists():
                return False
        if analyses.transition_counts.output.statistics:
            if not (output_dir / "transition_counts_statistics.pkl").exists():
                return False
        if analyses.transition_counts.output.ranks:
            if not (output_dir / "transition_counts_ranks.pkl").exists():
                return False

    # Check bootstrap outputs
    if bootstrap_enabled:
        if bootstrap_seed is None or n_bootstrap_samples is None:
            return False
        # Defensive check: ensure bootstrap config exists
        if analyses.bootstrap is None:
            return False

        # Check diff bootstrap if enabled
        if analyses.bootstrap.apply_to.diff and analyses.diff.enabled:
            bootstrap_samples_dir = (
                output_dir / "bootstrap" / "diff" / f"seed{bootstrap_seed}_n{n_bootstrap_samples}"
            )
            if not bootstrap_samples_dir.exists():
                return False
            # Check that all sample files exist
            for sample_idx in range(1, n_bootstrap_samples + 1):
                if not (bootstrap_samples_dir / f"{sample_idx}.pkl").exists():
                    return False

        # Check transition_counts bootstrap if enabled
        if (
            analyses.bootstrap.apply_to.transition_counts
            and analyses.transition_counts.enabled
        ):
            bootstrap_samples_dir = (
                output_dir
                / "bootstrap"
                / "transition_counts"
                / f"seed{bootstrap_seed}_n{n_bootstrap_samples}"
            )
            if not bootstrap_samples_dir.exists():
                return False
            # Check that all sample files exist
            for sample_idx in range(1, n_bootstrap_samples + 1):
                if not (bootstrap_samples_dir / f"{sample_idx}.pkl").exists():
                    return False

    return True


def process_combined_group(
    group: CombinedGroupConfig,
    filtering: FilteringConfig,
    analyses: AnalysesConfig,
    group_idx: int | None = None,
    total_groups: int | None = None,
) -> None:
    """
    Process a combined group of pairs sequentially to minimize memory usage.

    Processes one dataset at a time, one filter at a time, and one analysis at a time,
    releasing memory after each step.

    Args:
        group: Combined group configuration
        filtering: Filtering configuration
        analyses: Analyses configuration
        group_idx: Index of this group (for logging)
        total_groups: Total number of groups (for logging)
    """
    group_label = (
        f"[{group_idx}/{total_groups}]"
        if group_idx is not None and total_groups is not None
        else ""
    )

    logger.info(f"{group_label} Processing combined group: {group.name}")

    # Determine parent filter configurations
    parent_filter_configs = (
        filtering.parent_filters if filtering.parent_filters else [None]
    )

    # Process each list of pairs in group.pairs sequentially
    for dataset_idx, pairs_list in enumerate(group.pairs, 1):
        logger.info(
            f"{group_label} Processing dataset {dataset_idx}/{len(group.pairs)} "
            f"({len(pairs_list)} pair(s))"
        )

        # Skip empty pairs list
        if not pairs_list:
            raise ValueError(f"{group_label} Empty pairs list for dataset {dataset_idx}")

        # Check outputs for all filters BEFORE loading data
        bootstrap_enabled = (
            analyses.bootstrap is not None and analyses.bootstrap.enabled
        )
        bootstrap_seed = (
            analyses.bootstrap.config.seed if bootstrap_enabled else None
        )
        n_bootstrap_samples = (
            analyses.bootstrap.config.n_bootstrap_samples
            if bootstrap_enabled
            else None
        )

        # Determine which filters need processing
        filters_to_process = []
        
        # Get ngram_size from global analyses config for automatic subdirectory creation
        # ngram_size is always required and always added to path structure
        ngram_size = analyses.ngram_size
        
        for parent_filter in parent_filter_configs:
            filter_name = parent_filter.name if parent_filter else "no_filter"
            
            # Determine output directory for this filter
            output_dir = group.output_dir
            output_dir = output_dir / group.name
            # Always add ngram subdirectory (ngram_size is required in config)
            output_dir = output_dir / f"ngram{ngram_size}"
            if parent_filter is not None:
                if parent_filter.subdir is not None:
                    output_dir = output_dir / parent_filter.subdir / parent_filter.name
                else:
                    output_dir = output_dir / parent_filter.name
            output_dir.mkdir(parents=True, exist_ok=True)

            # Check if all outputs already exist
            if check_outputs_exist(
                output_dir=output_dir,
                analyses=analyses,
                bootstrap_enabled=bootstrap_enabled,
                bootstrap_seed=bootstrap_seed,
                n_bootstrap_samples=n_bootstrap_samples,
            ):
                logger.info(
                    f"{group_label} All outputs already exist for filter '{filter_name}', skipping..."
                )
            else:
                filters_to_process.append((parent_filter, output_dir))

        # If all filters are already processed, skip loading data entirely
        if not filters_to_process:
            logger.info(
                f"{group_label} All filters already processed for dataset {dataset_idx}, skipping data loading..."
            )
            continue

        # Combine pairs into single merged dataset (only for this dataset)
        # Column remapping is applied per-pair when loading files, then global match columns are used
        logger.info(
            f"{group_label} Loading data for dataset {dataset_idx} ({len(filters_to_process)} filter(s) need processing)..."
        )
        try:
            merged_df = combine_pairs_into_datasets(
                pairs=pairs_list,
                match_col_parents=analyses.match_col_parents,
                match_col_mutants=analyses.match_col_mutants,
            )
        except Exception as e:
            raise RuntimeError(f"{group_label} Error combining pairs for dataset {dataset_idx}: {e}") from e

        # Process only filters that need processing
        for parent_filter, output_dir in filters_to_process:
            filter_name = parent_filter.name if parent_filter else "no_filter"
            logger.info(f"{group_label} Applying parent filter: {filter_name}")

            # Apply parent filtering if configured
            filtered_merged_df = merged_df.copy()

            if parent_filter is not None:
                # Extract column filters as plain dicts
                column_filters = []
                for col_filter in parent_filter.columns:
                    filter_dict = {
                        "column": col_filter.column,
                        "dtype": col_filter.filter_config.dtype,
                    }
                    if col_filter.filter_config.dtype == "categorical":
                        filter_dict["values"] = col_filter.filter_config.values
                    elif col_filter.filter_config.dtype == "float":
                        filter_dict["min"] = col_filter.filter_config.min
                        filter_dict["max"] = col_filter.filter_config.max

                    column_filters.append(filter_dict)

                filtered_merged_df = filter_merged_parents(
                    merged_df=filtered_merged_df,
                    filter_name=parent_filter.name,
                    column_filters=column_filters,
                )

            # Compute diff DataFrame (using pre-merged dataframe)
            logger.info(f"{group_label} Computing diff DataFrame...")
            diff_df = compute_diff(
                merged_df=filtered_merged_df,
                value_cols=analyses.diff.config.value_cols,
                value_preprocessing=analyses.diff.config.value_preprocessing,
                add_relative=analyses.diff.config.add_relative,
            )
            logger.info(
                f"{group_label} Diff DataFrame: {len(diff_df)} rows, {len(diff_df.columns)} columns"
            )

            # Release filtered merged dataframe from memory
            del filtered_merged_df

            # Use mutation config from transition_counts (required for filtering and bootstrap)
            if not analyses.transition_counts.enabled:
                raise ValueError(
                    f"{group_label} transition_counts must be enabled. "
                    "Mutation config is required for filtering and bootstrap analysis."
                )
            mutation_config = analyses.transition_counts.config

            diff_df = apply_post_combination_filters(
                diff_df=diff_df,
                filter_config=filtering.post_combination_filters,
                mutation_config=mutation_config,
            )

            # Output directory already determined and checked above (in filters_to_process)

            # Use group name for dataset_name
            dataset_name = group.name

            # Run diff analysis sequentially and release memory
            if analyses.diff.enabled:
                run_diff_analysis(
                    diff_df=diff_df,
                    analysis_config=analyses.diff,
                    output_dir=output_dir,
                    dataset_name=dataset_name,
                    mutation_config=mutation_config if analyses.diff.output.ranks else None,
                    diff_config=analyses.diff.config if analyses.diff.output.ranks else None,
                    global_ngram_size=analyses.ngram_size,
                )

            # Run transition counts analysis sequentially
            aggregate_counter = None
            if analyses.transition_counts.enabled:
                aggregate_counter = run_transition_counts_analysis(
                    diff_df=diff_df,
                    analysis_config=analyses.transition_counts,
                    output_dir=output_dir,
                    dataset_name=dataset_name,
                    diff_config=analyses.diff.config if analyses.diff.enabled else None,
                    global_ngram_size=analyses.ngram_size,
                )

            # Run bootstrap analyses sequentially if enabled
            if analyses.bootstrap is not None and analyses.bootstrap.enabled:
                # Check if we need to run bootstrap for either analysis type
                need_diff_bootstrap = (
                    analyses.bootstrap.apply_to.diff and analyses.diff.enabled
                )
                need_transition_counts_bootstrap = (
                    analyses.bootstrap.apply_to.transition_counts
                    and analyses.transition_counts.enabled
                )

                if need_diff_bootstrap or need_transition_counts_bootstrap:
                    # For diff bootstrap, we need aggregate_counter and proper mutation_config
                    # If transition_counts is disabled, we need to ensure mutation_config is valid
                    if need_diff_bootstrap and not analyses.transition_counts.enabled:
                        raise ValueError(
                            "transition_counts analysis must be enabled for diff bootstrap. "
                            "Diff bootstrap requires mutation configuration (parent_col, mutant_col, ngram_size, etc.) "
                            "which is provided by transition_counts.config."
                        )

                    # For diff bootstrap, we need aggregate_counter
                    # Compute it if not already computed (if transition_counts is disabled)
                    diff_aggregate_counter = aggregate_counter
                    if need_diff_bootstrap and diff_aggregate_counter is None:
                        logger.info("Computing aggregate mutation counter for diff bootstrap...")
                        diff_aggregate_counter = compute_aggregate_mutation_counter(
                            diff_df=diff_df,
                            parent_col=mutation_config.parent_sequence_col,
                            mutant_col=mutation_config.mutant_sequence_col,
                            ngram_size=mutation_config.ngram_size,
                            allow_ngrams_overlap=mutation_config.allow_ngrams_overlap,
                        )

                    if need_transition_counts_bootstrap:
                        if aggregate_counter is None:
                            raise ValueError(
                                "aggregate_counter is required for transition_counts bootstrap. "
                                "It should be computed by transition_counts analysis."
                            )
                        if not analyses.diff.enabled:
                            raise ValueError(
                                "diff analysis must be enabled for transition_counts bootstrap "
                                "to determine value columns for statistics computation."
                            )

                    # Get seed from bootstrap config (can be overridden by command line)
                    bootstrap_seed = analyses.bootstrap.config.seed

                    # Generate bootstrap datasets once
                    logger.info(
                        f"Generating {analyses.bootstrap.config.n_bootstrap_samples} bootstrap datasets..."
                    )
                    bootstrap_dfs = list(
                        bootstrap_df_generator(
                            df=diff_df,
                            n_bootstrap_samples=analyses.bootstrap.config.n_bootstrap_samples,
                            bootstrap_group_cols=analyses.bootstrap.config.bootstrap_group_cols,
                            stratify_cols=analyses.bootstrap.config.stratify_cols,
                            bootstrap_frac=analyses.bootstrap.config.bootstrap_frac,
                            seed=bootstrap_seed,
                        )
                    )
                    logger.success(
                        f"Generated {len(bootstrap_dfs)} bootstrap datasets"
                    )

                    # Run bootstrap analysis for diff if needed
                    if need_diff_bootstrap:
                        logger.info("Running bootstrap analysis for diff...")
                        run_bootstrap_analysis_for_type(
                            bootstrap_dfs=bootstrap_dfs,
                            diff_df=diff_df,
                            aggregate_counter=diff_aggregate_counter,
                            analysis_type="diff",
                            mutation_config=mutation_config,
                            rank_config=analyses.diff.output.ranks_config or RankConfig(aggregation_func="mean"),
                            diff_config=analyses.diff.config,
                            output_dir=output_dir,
                            seed=bootstrap_seed,
                            global_ngram_size=analyses.ngram_size,
                        )

                    # Run bootstrap analysis for transition_counts if needed
                    if need_transition_counts_bootstrap:
                        logger.info("Running bootstrap analysis for transition_counts...")
                        run_bootstrap_analysis_for_type(
                            bootstrap_dfs=bootstrap_dfs,
                            diff_df=diff_df,
                            aggregate_counter=aggregate_counter,
                            analysis_type="transition_counts",
                            mutation_config=analyses.transition_counts.config,
                            rank_config=analyses.transition_counts.output.ranks_config or RankConfig(aggregation_func="mean"),
                            diff_config=analyses.diff.config,
                            output_dir=output_dir,
                            seed=bootstrap_seed,
                            global_ngram_size=analyses.ngram_size,
                        )

                    # Release bootstrap datasets from memory
                    del bootstrap_dfs

            # Release diff_df and aggregate_counter from memory after processing this filter
            del diff_df
            if aggregate_counter is not None:
                del aggregate_counter

        # Release merged dataset from memory after processing all filters for this dataset
        del merged_df

    logger.success(f"{group_label} Completed processing group: {group.name}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute bootstrap statistics for mutation analysis"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file or directory containing YAML files",
    )
    args = parser.parse_args()

    # Collect config files - automatically detect if path is a file or directory
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config path not found at {config_path}")
        sys.exit(1)

    config_paths = []
    if config_path.is_dir():
        # Find all YAML files in the directory
        config_paths = sorted(config_path.glob("*.yaml")) + sorted(
            config_path.glob("*.yml")
        )
        if not config_paths:
            logger.error(f"No YAML files found in {config_path}")
            sys.exit(1)
        logger.info(f"Found {len(config_paths)} config file(s) in {config_path}")
    elif config_path.is_file():
        # Single config file
        if not config_path.suffix.lower() in [".yaml", ".yml"]:
            logger.warning(
                f"Config file does not have .yaml or .yml extension: {config_path}"
            )
        config_paths = [config_path]
    else:
        logger.error(f"Config path is neither a file nor a directory: {config_path}")
        sys.exit(1)

    # Process each config file
    total_configs = len(config_paths)
    for config_idx, config_path in enumerate(config_paths, 1):
        logger.info(f"\n{'#'*60}")
        logger.info(f"Processing config {config_idx}/{total_configs}: {config_path.name}")
        logger.info(f"{'#'*60}")

        try:
            config = load_config(config_path)

            # Validate that combined_groups is provided
            if not config.combined_groups:
                logger.error("No combined_groups configuration provided")
                sys.exit(1)

            total_groups = len(config.combined_groups)
            logger.info(f"Processing {total_groups} combined group(s)")

            # Assert that all group names are unique
            group_names = [group.name for group in config.combined_groups]
            assert len(group_names) == len(set(group_names)), (
                f"Group names must be unique. Found duplicates: {[name for name in group_names if group_names.count(name) > 1]}"
            )

            # Process each combined group
            for group_idx, group in enumerate(config.combined_groups, 1):
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing group {group_idx}/{total_groups}: {group.name}")
                logger.info(f"{'='*60}")

                try:
                    process_combined_group(
                        group=group,
                        filtering=config.filtering,
                        analyses=config.analyses,
                        group_idx=group_idx,
                        total_groups=total_groups,
                    )
                    logger.success(f"[{group_idx}/{total_groups}] Completed successfully")
                except Exception as e:
                    logger.error(f"[{group_idx}/{total_groups}] Error: {e}")
                    raise

            logger.success(f"\n{'='*60}")
            logger.success(f"All {total_groups} group(s) processed successfully for {config_path.name}!")
            logger.success(f"{'='*60}")
        except Exception as e:
            logger.error(f"Error processing config {config_path.name}: {e}")
            raise

    logger.success(f"\n{'#'*60}")
    logger.success(f"All {total_configs} config file(s) processed successfully!")
    logger.success(f"{'#'*60}")


if __name__ == "__main__":
    main()
