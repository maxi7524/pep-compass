import sys
from pathlib import Path
from typing import Any
import argparse
import pandas as pd
import yaml
from pydantic import BaseModel, Field, model_validator
from loguru import logger
from tqdm import tqdm
from pep_compass.models.apex.APEX_predictor import PredictorAPEX


class PredictConfig(BaseModel):
    """Configuration for APEX batch predictions."""

    input_dir: Path | None = Field(
        default=None, description="Directory containing input CSV files"
    )
    input_files: list[Path] | None = Field(
        default=None, description="List of input CSV files to process"
    )
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

    @model_validator(mode='after')
    def validate_input_source(self) -> 'PredictConfig':
        """Validate that exactly one of input_dir or input_files is provided."""
        sources = [
            self.input_dir is not None,
            self.input_files is not None,
        ]
        if sum(sources) != 1:
            raise ValueError(
                "Exactly one of 'input_dir' or 'input_files' must be provided"
            )
        return self


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


def main():
    parser = argparse.ArgumentParser(
        description="Run APEX predictions on CSV file(s) using a YAML config. Can process a list of files or all CSVs in a directory."
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

    # Resolve paths
    output_dir = cfg.output_dir if cfg.output_dir.is_absolute() else (config_path.parent / cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    predictor = PredictorAPEX(**cfg.apex_predictor_kwargs)

    # Determine input files to process
    if cfg.input_files is not None:
        files = []
        for input_file in cfg.input_files:
            input_path = input_file if input_file.is_absolute() else (config_path.parent / input_file).resolve()
            if not input_path.exists():
                logger.error(f"Input file not found at {input_path}")
                sys.exit(1)
            files.append(input_path)
    else:
        input_path = cfg.input_dir if cfg.input_dir.is_absolute() else (config_path.parent / cfg.input_dir).resolve()
        if not input_path.exists():
            logger.error(f"Input directory not found at {input_path}")
            sys.exit(1)
        files = sorted(input_path.glob(cfg.file_glob))
        if not files:
            logger.error(f"No files matching '{cfg.file_glob}' found in {input_path}")
            sys.exit(1)

    for input_path in tqdm(files, desc="Processing files"):
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
