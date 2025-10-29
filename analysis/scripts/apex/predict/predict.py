import sys
from pathlib import Path
from typing import Any
import argparse
import pandas as pd
import yaml
from pydantic import BaseModel, Field
from loguru import logger

from pep_compass.models.apex.APEX_predictor import PredictorAPEX


class PredictConfig(BaseModel):
    """Configuration for APEX batch predictions."""

    input_dir: Path = Field(description="Directory containing input CSV files")
    output_dir: Path = Field(description="Directory to write output CSV files")
    file_glob: str = Field(
        default="*.csv", description="Glob pattern to select input files in input_dir"
    )
    sequence_column: str = Field(
        default="sequence", description="Column name with peptide sequences"
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
    for i, pathogen in enumerate(predictor.pathogen_list):
        df[pathogen] = preds[:, i]
    return df


def load_config(config_path: Path) -> PredictConfig:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return PredictConfig(**config_dict)


def main():
    parser = argparse.ArgumentParser(
        description="Run APEX predictions on all CSVs in a directory using a YAML config"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found at {config_path}")
        sys.exit(1)

    cfg = load_config(config_path)

    # Resolve relative paths against project root (same style as reference script)
    script_dir = Path(__file__).parent
    project_root = (script_dir / ".." / ".." / ".." / "..").resolve()

    input_dir = (
        cfg.input_dir if cfg.input_dir.is_absolute() else (project_root / cfg.input_dir)
    )
    output_dir = (
        cfg.output_dir
        if cfg.output_dir.is_absolute()
        else (project_root / cfg.output_dir)
    )

    if not input_dir.exists():
        logger.error(f"Input directory not found at {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    predictor = PredictorAPEX(**cfg.apex_predictor_kwargs)

    files = sorted(input_dir.glob(cfg.file_glob))
    if not files:
        logger.error(f"No files matching '{cfg.file_glob}' found in {input_dir}")
        sys.exit(1)

    for input_path in files:
        df = pd.read_csv(input_path)
        df = run_predictions_on_df(
            df,
            sequence_column=cfg.sequence_column,
            predictor=predictor,
            use_tqdm=cfg.use_tqdm,
        )
        out_path = output_dir / input_path.name
        df.to_csv(out_path, index=False)
        logger.info(f"Wrote: {out_path}")

    logger.success("All files processed successfully.")


if __name__ == "__main__":
    main()
