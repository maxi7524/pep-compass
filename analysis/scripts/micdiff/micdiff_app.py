#!/usr/bin/env python3
"""Compute bootstrap statistics for mutation analysis."""
import sys
from pathlib import Path
import pandas as pd
from loguru import logger
import pickle

from .config import Config, PairConfig, ParentFilterConfig, load_config
from .utils import compute_diff
from .mutations import compute_aggregate_mutation_counter
from .analysis import mutation_statistics_generator, compute_ranks_for_single_sample
from .filtering import filter_parents_and_mutants
from tqdm import tqdm


def process_single_pair(
    pair: PairConfig,
    config: Config,
    pair_idx: int | None = None,
    total_pairs: int | None = None,
    parent_filter_config: ParentFilterConfig | None = None,
) -> None:
    """Process a single parent-mutant pair."""
    pair_label = (
        f"[{pair_idx}/{total_pairs}]"
        if pair_idx is not None and total_pairs is not None
        else ""
    )

    filter_label = f"_{parent_filter_config.name}" if parent_filter_config else ""

    # Use paths directly (relative paths work relative to current working directory)
    parents_path = pair.parent
    if not parents_path.exists():
        logger.error(f"{pair_label} Parents dataset file not found at {parents_path}")
        return

    mutants_path = pair.mutant
    if not mutants_path.exists():
        logger.error(f"{pair_label} Mutants dataset file not found at {mutants_path}")
        return

    # Extract mutant name from path for naming
    mutant_name = Path(mutants_path).stem

    logger.info(f"{pair_label} Processing: {mutant_name}{filter_label}")
    logger.info(f"{pair_label} Loading parents dataset from: {parents_path}")
    parents_df = pd.read_csv(parents_path)
    logger.info(
        f"{pair_label} Parents dataset: {len(parents_df)} rows, {len(parents_df.columns)} columns"
    )

    logger.info(f"{pair_label} Loading mutants dataset from: {mutants_path}")
    mutants_df = pd.read_csv(mutants_path)
    logger.info(
        f"{pair_label} Mutants dataset: {len(mutants_df)} rows, {len(mutants_df.columns)} columns"
    )

    # Apply parent filtering if configured
    if parent_filter_config is not None:
        logger.info(f"{pair_label} Applying parent filter: {parent_filter_config.name}")

        # Extract column filters as plain dicts
        column_filters = []
        for col_filter in parent_filter_config.columns:
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

        parents_df, mutants_df = filter_parents_and_mutants(
            parents_df=parents_df,
            mutants_df=mutants_df,
            filter_name=parent_filter_config.name,
            column_filters=column_filters,
            match_col_parents=config.diff.match_col_parents,
            match_col_mutants=config.diff.match_col_mutants,
        )

    # Apply filtering if configured
    if any(
        [
            config.filter.filter_identities,
            config.filter.filter_length_mismatch,
            config.filter.deduplicate,
        ]
    ):
        logger.info(f"{pair_label} Applying mutation filters...")
        from utils import (
            filter_identities,
            filter_length_mismatches,
            deduplicate_mutations,
        )

        initial_count = len(mutants_df)

        if config.filter.filter_identities:
            mutants_df = filter_identities(
                mutants_df,
                parent_col=config.mutation.parent_col,
                mutant_col=config.mutation.mutant_col,
                log_progress=True,
            )

        if config.filter.filter_length_mismatch:
            mutants_df = filter_length_mismatches(
                mutants_df,
                parent_col=config.mutation.parent_col,
                mutant_col=config.mutation.mutant_col,
                log_progress=True,
            )

        if config.filter.deduplicate:
            mutants_df = deduplicate_mutations(
                mutants_df,
                parent_col=config.mutation.parent_col,
                mutant_col=config.mutation.mutant_col,
                keep="first",
                log_progress=True,
            )

        logger.info(
            f"{pair_label} After filtering: {len(mutants_df)} rows (removed {initial_count - len(mutants_df)})"
        )

    # Use output directory directly (relative paths work relative to current working directory)
    output_dir = pair.output_dir

    # Add filter subdirectory if filtering is applied
    if parent_filter_config is not None:
        output_dir = output_dir / parent_filter_config.name

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"{pair_label} Output directory: {output_dir}")

    # A. Compute diff DataFrame
    logger.info(f"{pair_label} Computing diff DataFrame...")
    diff_df = compute_diff(
        parents_df=parents_df,
        mutants_df=mutants_df,
        match_col_parents=config.diff.match_col_parents,
        match_col_mutants=config.diff.match_col_mutants,
        value_cols=config.diff.value_cols,
        value_preprocessing=config.diff.value_preprocessing,
        add_relative=config.diff.add_relative,
    )
    logger.info(
        f"{pair_label} Diff DataFrame: {len(diff_df)} rows, {len(diff_df.columns)} columns"
    )

    # Save diff_df
    diff_output_path = output_dir / f"{mutant_name}_diff.csv"
    diff_df.to_csv(diff_output_path, index=False)
    logger.success(f"{pair_label} Saved diff DataFrame to: {diff_output_path}")

    # B. Compute aggregate mutation counter
    logger.info(f"{pair_label} Computing aggregate mutation counter...")
    aggregate_counter = compute_aggregate_mutation_counter(
        diff_df=diff_df,
        parent_col=config.mutation.parent_col,
        mutant_col=config.mutation.mutant_col,
        ngram_size=config.mutation.ngram_size,
        allow_ngrams_overlap=config.mutation.allow_ngrams_overlap,
    )
    logger.info(
        f"{pair_label} Total unique mutations in aggregate: {len(aggregate_counter)}"
    )

    # C. Compute bootstrap sample ranks
    logger.info(f"{pair_label} Computing bootstrap sample ranks...")
    logger.info(
        f"{pair_label} Generating {config.bootstrap.n_bootstrap_samples} bootstrap samples..."
    )

    # Determine value columns to use (those with _diff suffix)
    value_preprocessing_suffix = (
        f"_{config.diff.value_preprocessing}" if config.diff.value_preprocessing else ""
    )
    diff_value_cols = [
        f"{col}{value_preprocessing_suffix}_diff" for col in config.diff.value_cols
    ]

    # Prepare seed string for filenames
    seed_str = (
        f"seed{config.bootstrap.seed}"
        if config.bootstrap.seed is not None
        else "seedNone"
    )

    # Compute ranks for the full dataset (no bootstrap)
    logger.info(f"{pair_label} Computing ranks for full dataset...")
    from .analysis import compute_mutation_statistics_from_df

    full_stats = compute_mutation_statistics_from_df(
        diff_df=diff_df,
        aggregate_counter=aggregate_counter,
        parent_col=config.mutation.parent_col,
        mutant_col=config.mutation.mutant_col,
        value_cols=diff_value_cols,
        ngram_size=config.mutation.ngram_size,
        allow_ngrams_overlap=config.mutation.allow_ngrams_overlap,
    )

    full_dataset_ranks = compute_ranks_for_single_sample(
        stats_dict=full_stats,
        aggregate_cols=diff_value_cols,
        aggregate_counter=aggregate_counter,
        aggregation_func=config.rank.aggregation_func,
    )

    # Save full dataset ranks
    full_dataset_output_file = output_dir / f"{mutant_name}_{seed_str}_full_dataset.pkl"
    with open(full_dataset_output_file, "wb") as f:
        pickle.dump(full_dataset_ranks, f)
    logger.success(
        f"{pair_label} Saved full dataset ranks to: {full_dataset_output_file}"
    )

    stats_gen_ranks = mutation_statistics_generator(
        diff_df=diff_df,
        aggregate_counter=aggregate_counter,
        parent_col=config.mutation.parent_col,
        mutant_col=config.mutation.mutant_col,
        value_cols=diff_value_cols,
        ngram_size=config.mutation.ngram_size,
        allow_ngrams_overlap=config.mutation.allow_ngrams_overlap,
        n_bootstrap_samples=config.bootstrap.n_bootstrap_samples,
        bootstrap_group_cols=config.bootstrap.bootstrap_group_cols,
        stratify_cols=config.bootstrap.stratify_cols,
        bootstrap_frac=config.bootstrap.bootstrap_frac,
        seed=config.bootstrap.seed,
    )

    # Create subdirectory for bootstrap samples
    bootstrap_samples_dir = output_dir / f"{mutant_name}_{seed_str}_bootstrap_samples"
    bootstrap_samples_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"{pair_label} Bootstrap samples directory: {bootstrap_samples_dir}")

    # Process bootstrap samples with progress bar
    all_mutation_keys = list(aggregate_counter.keys())

    with tqdm(
        total=config.bootstrap.n_bootstrap_samples,
        desc=f"{pair_label} Bootstrap samples",
        unit="sample",
        ncols=100,
        leave=True,
        position=0,
    ) as pbar:
        for sample_idx, stats in enumerate(stats_gen_ranks, 1):
            # Compute ranks for this single sample
            sample_ranks = compute_ranks_for_single_sample(
                stats_dict=stats,
                aggregate_cols=diff_value_cols,
                aggregate_counter=aggregate_counter,
                aggregation_func=config.rank.aggregation_func,
            )

            # Save this bootstrap sample's ranks to a separate file (numbered without leading zeros)
            sample_file = bootstrap_samples_dir / f"{sample_idx}.pkl"
            with open(sample_file, "wb") as f:
                pickle.dump(sample_ranks, f)

            pbar.update(1)
            pbar.refresh()  # Force refresh to update display

    logger.success(
        f"{pair_label} Saved {config.bootstrap.n_bootstrap_samples} bootstrap sample files to {bootstrap_samples_dir}"
    )
    logger.info(f"{pair_label} Files for {len(all_mutation_keys)} mutations")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute bootstrap statistics for mutation analysis"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found at {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    # Determine pairs to process
    if config.pairs is not None:
        pairs_to_process = [
            PairConfig(
                parent=pair.parent,
                mutant=pair.mutant,
                output_dir=pair.output_dir,
            )
            for pair in config.pairs
        ]
    elif config.io is not None:
        # Convert single io config to a pair for backward compatibility
        pairs_to_process = [
            PairConfig(
                parent=config.io.parents_dataset_path,
                mutant=config.io.mutants_dataset_path,
                output_dir=config.io.output_dir,
            )
        ]
    else:
        logger.error("No pairs or io configuration provided")
        sys.exit(1)

    # Determine parent filter configurations
    parent_filter_configs = config.parent_filters if config.parent_filters else [None]

    total_pairs = len(pairs_to_process)
    total_filters = len(parent_filter_configs)
    total_combinations = total_pairs * total_filters

    logger.info(
        f"Processing {total_combinations} combination(s) "
        f"({total_pairs} pair(s) × {total_filters} filter(s))"
    )

    # Process each combination
    combination_idx = 0
    for pair_idx, pair in enumerate(pairs_to_process, 1):
        for filter_idx, parent_filter in enumerate(parent_filter_configs, 1):
            combination_idx += 1

            filter_name = parent_filter.name if parent_filter else "no_filter"
            logger.info(f"\n{'='*60}")
            logger.info(
                f"Processing combination {combination_idx}/{total_combinations}"
            )
            logger.info(f"Pair: {pair_idx}/{total_pairs}")
            logger.info(f"Filter: {filter_name} ({filter_idx}/{total_filters})")
            logger.info(f"{'='*60}")

            try:
                process_single_pair(
                    pair=pair,
                    config=config,
                    pair_idx=pair_idx,
                    total_pairs=total_pairs,
                    parent_filter_config=parent_filter,
                )
                logger.success(
                    f"[{combination_idx}/{total_combinations}] Completed successfully"
                )
            except Exception as e:
                logger.error(f"[{combination_idx}/{total_combinations}] Error: {e}")
                raise

    logger.success(f"\n{'='*60}")
    logger.success(f"All {total_combinations} combination(s) processed successfully!")
    logger.success(f"{'='*60}")


if __name__ == "__main__":
    main()
