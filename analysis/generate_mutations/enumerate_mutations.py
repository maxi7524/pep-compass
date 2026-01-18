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
) -> pd.DataFrame:
    """Generate mutants from single-position mutations for all sequences in a DataFrame."""
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
            ]
            new_rows.append(new_row)
    return pd.DataFrame(
        new_rows,
        columns=[
            "mutant",
            "position",
            "parent",
        ],
    )


class Config(BaseModel):
    """Configuration model for mutation enumeration."""

    input_file: Path = Field(description="Path to input CSV file containing peptides")
    output_path: Path = Field(description="Path to output CSV file")
    csv_separator: str = Field(
        default=",",
        description="Separator/delimiter for the input CSV file (e.g., ',', '\t', ';')",
    )
    seq_col: str = Field(
        default="Sequence",
        description="Name of the column containing peptide sequences",
    )
    direction_threshold: float = Field(
        default=0.0001,
        description="Direction significance threshold for mutation enumeration",
    )
    token_threshold: float = Field(
        default=0.05,
        description="Token threshold for mutation probability per position",
    )
    jacobian_mode: str = Field(
        default="approx", description="Jacobian computation mode: 'strict' or 'approx'"
    )
    jacobian_eps: float = Field(
        default=1e-6,
        description="Epsilon value for jacobian computation",
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

    if not config.input_file.exists():
        logger.error(f"Dataset file not found at {config.input_file}")
        sys.exit(1)

    logger.info(f"Loading dataset from: {config.input_file}")
    df = pd.read_csv(config.input_file, sep=config.csv_separator)
    logger.info(f"Total sequences loaded: {len(df)}")

    mutants_df = get_mutants_from_single_position_mutations_from_df(
        df=df,
        seq_col=config.seq_col,
        hydramp=hydramp,
        direction_significance_threshold=config.direction_threshold,
        token_threshold=config.token_threshold,
    )

    logger.info(f"Total mutants generated: {len(mutants_df)}")
    logger.info(f"Unique mutants: {mutants_df['mutant'].nunique()}")

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results to: {config.output_path}")
    mutants_df.to_csv(config.output_path, index=False)
    logger.success("Complete!")


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
        logger.error(f"Config path not found at {config_path}")
        sys.exit(1)

    if not config_path.is_file():
        logger.error(f"Config path is not a file: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    process_single_config(config)


if __name__ == "__main__":
    main()
