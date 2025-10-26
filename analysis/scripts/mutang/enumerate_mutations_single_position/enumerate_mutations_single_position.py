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


def load_hydramp_model(
    weights_dir: Path,
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
    encoder_path = weights_dir / "encoder_weights.pickle"
    decoder_path = weights_dir / "decoder_weights.pickle"
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder weights not found at {encoder_path}")
    if not decoder_path.exists():
        raise FileNotFoundError(f"Decoder weights not found at {decoder_path}")
    hydramp.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    hydramp.decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    logger.success("Model loaded successfully")
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
                direction_significance_threshold,
                token_threshold,
            ] + [getattr(row, col) for col in parent_keep_columns_rename_map.keys()]
            new_rows.append(new_row)
    return pd.DataFrame(
        new_rows,
        columns=[
            "mutant",
            "position",
            "parent",
            "direction_significance_threshold",
            "token_threshold",
        ]
        + list(parent_keep_columns_rename_map.values()),
    )


class Config(BaseModel):
    """Configuration model for mutation enumeration."""

    dataset_path: Path = Field(
        description="Path to input CSV file containing peptide sequences"
    )
    output_dir: Path = Field(description="Directory to save output CSV files")
    weights_dir: Path = Field(
        description="Directory containing encoder_weights.pickle and decoder_weights.pickle"
    )
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
    parent_keep_columns_rename_map: dict[str, str] = Field(
        default_factory=dict,
        description="Map parent dataset columns to output column names",
    )


def load_config(config_path: Path) -> Config:
    """Load and validate configuration from YAML file."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Enumerate single-position mutations from peptide sequences"
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

    script_dir = Path(__file__).parent
    project_root = (script_dir / ".." / ".." / ".." / "..").resolve()
    logger.info(f"Project root: {project_root}")

    weights_dir = config.weights_dir
    if not weights_dir.is_absolute():
        weights_dir = project_root / weights_dir
    logger.info(f"Weights directory: {weights_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hydramp = load_hydramp_model(
        weights_dir, jacobian_mode=config.jacobian_mode, device=device
    )

    dataset_path = config.dataset_path
    if not dataset_path.is_absolute():
        dataset_path = project_root / dataset_path
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

    # Get output directory
    output_dir = config.output_dir
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
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

        # Generate filename based on input name + parameters
        output_filename = f"{input_stem}_direction_threshold={d_thresh}_token_threshold={t_thresh}_jacobian_mode={config.jacobian_mode}.csv"
        output_path = output_dir / output_filename

        logger.info(f"Saving results to: {output_path}")
        mutants_df.to_csv(output_path, index=False)

    logger.success("All combinations complete!")


if __name__ == "__main__":
    main()
