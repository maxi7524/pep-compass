import sys
from pathlib import Path
import argparse

import pandas as pd
import yaml
from pydantic import BaseModel, Field
from loguru import logger
from tqdm import tqdm
from peptides import Peptide
from Bio.SeqUtils.ProtParam import ProteinAnalysis


class PredictConfig(BaseModel):
    """Configuration for physicochemical property predictions on a single CSV file."""

    input_file: Path = Field(description="Input CSV file to process")
    output_path: Path = Field(description="Output CSV file path")
    sequence_column: str = Field(
        default="sequence", description="Column name with peptide sequences"
    )
    csv_separator: str = Field(
        default=",",
        description="Separator/delimiter for input CSV/TSV files (e.g., ',', '\\t', ';')",
    )
    ph: float = Field(
        default=7.0,
        description="pH value used for charge calculation",
    )
    hydrophobicity_scale: str = Field(
        default="Aboderin",
        description="Hydrophobicity scale (passed to peptides.Peptide.hydrophobicity)",
    )
    use_tqdm: bool = Field(
        default=False, description="Show a tqdm progress bar over sequences"
    )


def compute_properties(
    sequence: str,
    ph: float = 7.0,
    hydrophobicity_scale: str = "Aboderin",
) -> dict[str, float]:
    """Compute all physicochemical properties for a single sequence."""
    bio = ProteinAnalysis(sequence)
    pep = Peptide(sequence)
    return {
        "charge": bio.charge_at_pH(ph),
        "hydrophobicity": pep.hydrophobicity(scale=hydrophobicity_scale),
        "isoelectric_point": bio.isoelectric_point(),
        "molecular_weight": bio.molecular_weight(),
        "instability_index": bio.instability_index(),
        "gravy": bio.gravy(),
        "aliphatic_index": pep.aliphatic_index(),
        "boman_index": pep.boman(),
    }


def run(config: PredictConfig):
    """Load CSV, compute properties, write output."""
    if not config.input_file.exists():
        logger.error(f"Input file not found: {config.input_file}")
        sys.exit(1)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(config.input_file, sep=config.csv_separator)
    logger.info(f"Loaded {len(df)} rows from {config.input_file}")

    if config.sequence_column not in df.columns:
        logger.error(f"Column '{config.sequence_column}' not found")
        sys.exit(1)

    sequences = df[config.sequence_column].astype(str).tolist()
    it = tqdm(sequences, desc="Computing properties") if config.use_tqdm else sequences

    rows = [
        compute_properties(seq, config.ph, config.hydrophobicity_scale) for seq in it
    ]
    prop_df = pd.DataFrame(rows, index=df.index)

    df = pd.concat([df, prop_df], axis=1)
    df.to_csv(config.output_path, index=False)
    logger.info(f"Wrote: {config.output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute physicochemical properties from a YAML config."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        cfg = PredictConfig(**yaml.safe_load(f))

    run(cfg)
    logger.success("Done.")


if __name__ == "__main__":
    main()
