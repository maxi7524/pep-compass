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
import sys
from pathlib import Path
from typing import Any, Optional
import argparse
import pandas as pd
import yaml
from pydantic import BaseModel, Field, model_validator
from loguru import logger
from tqdm import tqdm
from pep_compass.models.apex.APEX_predictor import PredictorAPEX
from utils.filtering import filter_by_length


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
    csv_separator: str = Field(
        default=",", description="Separator/delimiter for input CSV/TSV files (e.g., ',', '\t', ';')"
    )
    max_sequence_length: Optional[int] = Field(
        default=None,
        description="Maximum sequence length to process (inclusive). Sequences with length <= max_sequence_length are kept. If None, no length filtering is applied.",
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


def process_single_config(config: PredictConfig, config_path: Path):
    """Process a single configuration."""
    # Use paths directly (relative paths work relative to current working directory)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    predictor = PredictorAPEX(**config.apex_predictor_kwargs)

    # Determine input files to process
    if config.input_files is not None:
        files = []
        for input_file in config.input_files:
            if not input_file.exists():
                logger.error(f"Input file not found at {input_file}")
                sys.exit(1)
            files.append(input_file)
    else:
        input_path = config.input_dir
        if not input_path.exists():
            logger.error(f"Input directory not found at {input_path}")
            sys.exit(1)
        files = sorted(input_path.glob(config.file_glob))
        if not files:
            logger.error(f"No files matching '{config.file_glob}' found in {input_path}")
            sys.exit(1)

    for input_path in tqdm(files, desc="Processing files"):
        df = pd.read_csv(input_path, sep=config.csv_separator)
        
        # Filter by sequence length if specified (before processing)
        if config.max_sequence_length is not None:
            logger.info(f"Filtering sequences by length (<= {config.max_sequence_length})")
            df = filter_by_length(
                df,
                seq_col=config.sequence_column,
                max_length=config.max_sequence_length,
                log_progress=True,
            )
        
        df = run_predictions_on_df(
            df,
            sequence_column=config.sequence_column,
            predictor=predictor,
            use_tqdm=config.use_tqdm,
        )
        
        # Generate output filename with length filter suffix if applied
        input_stem = input_path.stem
        length_filter_suffix = ""
        if config.max_sequence_length is not None:
            length_filter_suffix = f"_leq{config.max_sequence_length}"
        
        output_filename = f"{input_stem}{length_filter_suffix}.csv"
        out_path = output_dir / output_filename
        df.to_csv(out_path, index=False)
        logger.info(f"Wrote: {out_path}")

    logger.success("All files processed successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Run APEX predictions on CSV file(s) using a YAML config. Can process a list of files or all CSVs in a directory."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file or directory containing YAML config files"
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
        cfg = load_config(config_file)
        process_single_config(cfg, config_file)
        logger.success(f"Completed config {idx}/{total_configs}: {config_file.name}")

    logger.success(f"\nAll {total_configs} config(s) processed successfully!")


if __name__ == "__main__":
    main()
