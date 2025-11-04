#!/usr/bin/env python3
"""Standalone script to compute diff dataframes from parent and mutant datasets."""
import sys
from pathlib import Path
import pandas as pd
from loguru import logger

from .config import ComputeDiffConfig, load_compute_diff_config
from .utils import compute_diff


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute diff DataFrame from parent and mutant datasets"
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

    config = load_compute_diff_config(config_path)

    # Find project root: analysis/scripts/micdiff/ -> project root (3 levels up)
    script_dir = Path(__file__).parent
    project_root = (script_dir / ".." / ".." / "..").resolve()

    # Resolve paths
    parents_path = config.parents_dataset_path
    if not parents_path.is_absolute():
        parents_path = project_root / parents_path
    if not parents_path.exists():
        logger.error(f"Parents dataset file not found at {parents_path}")
        sys.exit(1)

    mutants_path = config.mutants_dataset_path
    if not mutants_path.is_absolute():
        mutants_path = project_root / mutants_path
    if not mutants_path.exists():
        logger.error(f"Mutants dataset file not found at {mutants_path}")
        sys.exit(1)

    output_path = config.output_path
    if not output_path.is_absolute():
        output_path = project_root / output_path

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading parents dataset from: {parents_path}")
    parents_df = pd.read_csv(parents_path)
    logger.info(
        f"Parents dataset: {len(parents_df)} rows, {len(parents_df.columns)} columns"
    )

    logger.info(f"Loading mutants dataset from: {mutants_path}")
    mutants_df = pd.read_csv(mutants_path)
    logger.info(
        f"Mutants dataset: {len(mutants_df)} rows, {len(mutants_df.columns)} columns"
    )

    # Compute diff DataFrame
    logger.info("Computing diff DataFrame...")
    diff_df = compute_diff(
        parents_df=parents_df,
        mutants_df=mutants_df,
        match_col_parents=config.diff.match_col_parents,
        match_col_mutants=config.diff.match_col_mutants,
        value_cols=config.diff.value_cols,
        value_preprocessing=config.diff.value_preprocessing,
        add_relative=config.diff.add_relative,
    )
    logger.info(f"Diff DataFrame: {len(diff_df)} rows, {len(diff_df.columns)} columns")

    # Save diff_df
    diff_df.to_csv(output_path, index=False)
    logger.success(f"Saved diff DataFrame to: {output_path}")

    logger.success("Diff computation complete!")


if __name__ == "__main__":
    main()
