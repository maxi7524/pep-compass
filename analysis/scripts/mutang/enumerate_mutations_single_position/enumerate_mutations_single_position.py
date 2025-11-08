#!/usr/bin/env python3
"""Enumerate single-position mutations from peptide sequences using HydrAMP."""
import sys
from pathlib import Path
from typing import Generator
import pandas as pd
import torch
import yaml
from pydantic import BaseModel, Field
from tqdm import tqdm
from loguru import logger
from pep_compass.local_enumeration.mutation.mutation_enumerator import (
    MutationEnumerationInTangentSpace,
)
from pep_compass.models.encoder_decoder.hydramp_encoder_decoder import (
    HydrAMPEncoderDecoder,
)

from utils.filtering import (
    filter_identities,
    filter_length_mismatches,
    deduplicate_mutations,
)


def load_hydramp_model(
    jacobian_mode: str = "approx",
    jacobian_eps: float = 1e-6,
    field_eps: float = 1e-6,
    device: torch.device = None,
) -> HydrAMPEncoderDecoder:
    """Load and initialize the HydrAMP encoder-decoder model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    hydramp = HydrAMPEncoderDecoder(
        jacobian_mode=jacobian_mode,
        jacobian_eps=jacobian_eps,
        field_eps=field_eps,
        device=device,
    )
    logger.success("Model initialized")
    return hydramp


def get_mutants_from_single_position_mutations(
    seq: str,
    single_positions_mutations: dict[int, list[int]],
    alphabet: list[str] = list(" ACDEFGHIKLMNPQRSTVWY"),
) -> Generator[tuple[int, str], None, None]:
    """Generate mutants by applying single-position mutations."""
    for pos, mutations in single_positions_mutations.items():
        for mut in mutations:
            yield pos, seq[:pos] + alphabet[mut] + seq[pos + 1 :]


def process_peptide_mutations(
    seq: str,
    hydramp: HydrAMPEncoderDecoder,
    mutang: MutationEnumerationInTangentSpace,
) -> list[tuple[int, str]]:
    """Process a single peptide to generate mutations."""
    z = hydramp.encode_peptides([seq])
    jac = hydramp.decoder_jacobian(z)
    U, S, _ = torch.linalg.svd(jac, full_matrices=False)
    mutations = mutang.get_mutations_from_s_u(
        S[0].detach().cpu().numpy(), U[0].detach().cpu().numpy()
    )
    return list(get_mutants_from_single_position_mutations(seq, mutations))


def get_mutants_from_single_position_mutations_from_df(
    df: pd.DataFrame,
    seq_col: str,
    hydramp: HydrAMPEncoderDecoder,
    direction_significance_threshold: float,
    token_threshold: float,
    parent_keep_columns_rename_map: dict[str, str] = None,
) -> pd.DataFrame:
    """Generate mutants from single-position mutations for all sequences in a DataFrame."""
    if parent_keep_columns_rename_map is None:
        parent_keep_columns_rename_map = {}
    mutang = MutationEnumerationInTangentSpace(
        direction_significance_threshold=direction_significance_threshold,
        token_threshold=token_threshold,
    )
    new_rows = []
    for row in tqdm(df.itertuples(), total=len(df), desc="Processing peptides"):
        seq = getattr(row, seq_col)
        mutants = process_peptide_mutations(seq, hydramp, mutang)
        for pos, mutant in mutants:
            new_row = [
                mutant,
                pos,
                seq,
            ] + [getattr(row, col) for col in parent_keep_columns_rename_map.keys()]
            new_rows.append(new_row)
    return pd.DataFrame(
        new_rows,
        columns=[
            "mutant",
            "position",
            "parent",
        ]
        + list(parent_keep_columns_rename_map.values()),
    )


class Config(BaseModel):
    """Configuration model for mutation enumeration."""

    dataset_path: Path = Field(
        description="Path to input CSV file containing peptide sequences"
    )
    output_dir: Path = Field(description="Directory to save output CSV files")
    seq_col: str = Field(
        default="Sequence",
        description="Name of the column containing peptide sequences",
    )
    direction_threshold: float | list[float] = Field(
        default=0.0001,
        description="Direction significance threshold(s) for mutation enumeration",
    )
    token_threshold: float | list[float] = Field(
        default=0.05,
        description="Token threshold(s) for mutation probability per position",
    )
    jacobian_mode: str = Field(
        default="approx", description="Jacobian computation mode: 'strict' or 'approx'"
    )
    jacobian_eps: float = Field(
        default=1e-6,
        description="Epsilon value for jacobian computation",
    )
    parent_keep_columns_rename_map: dict[str, str] = Field(
        default_factory=dict,
        description="Map parent dataset columns to output column names",
    )
    deduplicate: bool = Field(
        default=True,
        description="Remove duplicate rows where both parent and mutant sequences are the same",
    )
    filter_identities: bool = Field(
        default=True,
        description="Filter out cases where parent sequence equals mutant sequence",
    )
    filter_length_mismatch: bool = Field(
        default=True,
        description="Filter out cases where parent and mutant have different lengths after stripping whitespace",
    )


def load_config(config_path: Path) -> Config:
    """Load and validate configuration from YAML file."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)


def process_single_config(config: Config):
    """Process a single configuration."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hydramp = load_hydramp_model(
        jacobian_mode=config.jacobian_mode,
        jacobian_eps=config.jacobian_eps,
        device=device,
    )

    # Use dataset path directly (relative paths work relative to current working directory)
    dataset_path = config.dataset_path
    if not dataset_path.exists():
        logger.error(f"Dataset file not found at {dataset_path}")
        sys.exit(1)

    logger.info(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    logger.info(f"Total sequences to process: {len(df)}")

    # Normalize parameters to lists
    direction_thresholds = (
        [config.direction_threshold]
        if isinstance(config.direction_threshold, float)
        else config.direction_threshold
    )
    token_thresholds = (
        [config.token_threshold]
        if isinstance(config.token_threshold, float)
        else config.token_threshold
    )

    # Validate that both lists have the same length
    if len(direction_thresholds) != len(token_thresholds):
        logger.error(
            f"direction_threshold and token_threshold lists must have the same length. Got {len(direction_thresholds)} and {len(token_thresholds)}"
        )
        sys.exit(1)

    # Generate parameter pairs (by index, not combinations)
    param_pairs = list(zip(direction_thresholds, token_thresholds))
    logger.info(f"Running {len(param_pairs)} parameter pairs")

    # Use output directory directly (relative paths work relative to current working directory)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get base filename from input dataset
    input_stem = dataset_path.stem

    for i, (d_thresh, t_thresh) in enumerate(param_pairs, 1):
        logger.info(
            f"\n=== Pair {i}/{len(param_pairs)}: d={d_thresh}, t={t_thresh} ==="
        )

        mutants_df = get_mutants_from_single_position_mutations_from_df(
            df=df,
            seq_col=config.seq_col,
            hydramp=hydramp,
            direction_significance_threshold=d_thresh,
            token_threshold=t_thresh,
            parent_keep_columns_rename_map=config.parent_keep_columns_rename_map,
        )

        logger.info(f"Total mutants generated: {len(mutants_df)}")
        logger.info(f"Unique mutants: {mutants_df['mutant'].nunique()}")

        

        if config.filter_identities:
            mutants_df = filter_identities(
                mutants_df, parent_col="parent", mutant_col="mutant", log_progress=True
            )

        if config.filter_length_mismatch:
            mutants_df = filter_length_mismatches(
                mutants_df, parent_col="parent", mutant_col="mutant", log_progress=True
            )

        if config.deduplicate:
            mutants_df = deduplicate_mutations(
                mutants_df,
                parent_col="parent",
                mutant_col="mutant",
                keep="first",
                log_progress=True,
            )

        # Generate filename based on input name + parameters
        output_filename = f"{input_stem}_direction_threshold={d_thresh}_token_threshold={t_thresh}_jacobian_mode={config.jacobian_mode}_jacobian_eps={config.jacobian_eps}.csv"
        output_path = output_dir / output_filename

        logger.info(f"Saving results to: {output_path}")
        mutants_df.to_csv(output_path, index=False)

    logger.success("All combinations complete!")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Enumerate single-position mutations from peptide sequences"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file or directory containing YAML config files",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config path not found at {config_path}")
        sys.exit(1)

    # Determine if it's a file or directory
    if config_path.is_file():
        # Single config file
        config_files = [config_path]
    elif config_path.is_dir():
        # Directory with configs - find all YAML files
        config_files = sorted(config_path.glob("*.yaml")) + sorted(
            config_path.glob("*.yml")
        )
        if not config_files:
            logger.error(f"No YAML config files found in directory {config_path}")
            sys.exit(1)
        logger.info(f"Found {len(config_files)} config file(s) in directory")
    else:
        logger.error(f"Config path is neither a file nor a directory: {config_path}")
        sys.exit(1)

    # Process each config sequentially
    total_configs = len(config_files)
    for idx, config_file in enumerate(config_files, 1):
        logger.info(
            f"\n{'='*80}\n"
            f"Processing config {idx}/{total_configs}: {config_file.name}\n"
            f"{'='*80}"
        )
        config = load_config(config_file)
        process_single_config(config)
        logger.success(f"Completed config {idx}/{total_configs}: {config_file.name}")

    logger.success(f"\nAll {total_configs} config(s) processed successfully!")


if __name__ == "__main__":
    main()
