import sys
from pathlib import Path
from typing import Any
import argparse
import pandas as pd
import yaml
from pydantic import BaseModel, Field
from loguru import logger
from tqdm import tqdm
from pep_compass.models.apex.APEX_predictor import PredictorAPEX


class PredictConfig(BaseModel):
    """Configuration for APEX predictions on a single CSV file."""

    input_file: Path = Field(description="Input CSV file to process")
    output_path: Path = Field(description="Output CSV file path")
    sequence_column: str = Field(
        default="sequence", description="Column name with peptide sequences"
    )
    csv_separator: str = Field(
        default=",",
        description="Separator/delimiter for input CSV/TSV files (e.g., ',', '\\t', ';')",
    )
    apex_predictor_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Keyword arguments for PredictorAPEX"
    )
    use_tqdm: bool = Field(
        default=False, description="Show a tqdm progress bar over sequences"
    )


def run_predictions_on_df(
    df: pd.DataFrame, sequence_column: str, predictor: PredictorAPEX, use_tqdm: bool
) -> pd.DataFrame:
    if sequence_column not in df.columns:
        raise ValueError(f"Column '{sequence_column}' not found in input dataframe")
    sequences = df[sequence_column].astype(str).tolist()
    preds = predictor.predict(sequences, use_tqdm=use_tqdm)
    # Vectorized assignment: create a DataFrame from predictions and assign all columns at once
    pred_df = pd.DataFrame(preds, columns=predictor.pathogen_list, index=df.index)
    return pd.concat([df, pred_df], axis=1)


def load_config(config_path: Path) -> PredictConfig:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return PredictConfig(**config_dict)


def process_single_config(config: PredictConfig):
    """Process a single configuration."""
    if not config.input_file.exists():
        logger.error(f"Input file not found at {config.input_file}")
        sys.exit(1)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    predictor = PredictorAPEX(**config.apex_predictor_kwargs)

    df = pd.read_csv(config.input_file, sep=config.csv_separator)

    df = run_predictions_on_df(
        df,
        sequence_column=config.sequence_column,
        predictor=predictor,
        use_tqdm=config.use_tqdm,
    )

    df.to_csv(config.output_path, index=False)
    logger.info(f"Wrote: {config.output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run APEX predictions on a single CSV file using a YAML config."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config path not found at {config_path}")
        sys.exit(1)

    if not config_path.is_file():
        logger.error(f"Config path is not a file: {config_path}")
        sys.exit(1)

    cfg = load_config(config_path)
    process_single_config(cfg)
    logger.success("Completed config.")


if __name__ == "__main__":
    main()
