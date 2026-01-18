#!/usr/bin/env python3
"""Merge multiple delimited files into a single CSV with source and fixed columns."""

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pydantic import BaseModel, ValidationError, field_validator, model_validator


class Config(BaseModel):
    path: Path
    sep: str
    fixed_columns: dict[str, Any]

    @field_validator("path")
    @classmethod
    def _path_exists(cls, value: Path) -> Path:
        if not value.exists():
            raise ValueError(f"Input file not found: {value}")
        return value


class MergeConfig(BaseModel):
    output_path: Path
    source_col: str
    inputs: list[Config]
    rename_map: dict[str, str] | None
    id_col: str | None
    keep_columns: list[str] | None
    seq_col: str
    min_length: int | None
    max_length: int | None
    deduplicate: bool
    dedup_on: list[str]
    merge_separator: str

    @field_validator("inputs")
    @classmethod
    def _inputs_non_empty(cls, value: list[Config]) -> list[Config]:
        if not value:
            raise ValueError("inputs must be a non-empty list.")
        return value

    @model_validator(mode="after")
    def _validate_rename_map(self) -> "MergeConfig":
        if self.rename_map is None:
            raise ValueError("rename_map must be provided (can be empty).")
        if self.keep_columns is None:
            raise ValueError("keep_columns must be provided (can be empty).")
        if self.id_col is None:
            raise ValueError("id_col must be provided (can be null).")
        if self.min_length is None and self.max_length is None:
            raise ValueError("min_length or max_length must be provided (can be null).")
        return self


def load_config(path: Path) -> MergeConfig:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping at the top level.")
    return MergeConfig.model_validate(config)


def load_single_input(
    input_cfg: Config,
    source_col: str,
    rename_map: dict[str, str],
    id_col: str | None,
    keep_columns: list[str] | None,
    seq_col: str,
    min_length: int | None,
    max_length: int | None,
) -> pd.DataFrame:
    input_path = input_cfg.path
    df = pd.read_csv(input_path, sep=input_cfg.sep)

    df[source_col] = input_path.name

    df = df.rename(columns=rename_map)

    if id_col and id_col != "id" and "id" not in df.columns:
        df = df.rename(columns={id_col: "id"})

    if min_length is not None or max_length is not None:
        seq_lengths = df[seq_col].astype(str).str.len()
        if min_length is not None:
            df = df[seq_lengths >= min_length]
        if max_length is not None:
            df = df[seq_lengths <= max_length]

    for key, value in input_cfg.fixed_columns.items():
        df[key] = value

    if keep_columns:
        df = df[keep_columns]

    return df


def merge_sources(config_path: Path) -> None:
    config = load_config(config_path)

    frames = [
        load_single_input(
            input_cfg=input_cfg,
            source_col=config.source_col,
            rename_map=config.rename_map,
            id_col=config.id_col,
            keep_columns=config.keep_columns,
            seq_col=config.seq_col,
            min_length=config.min_length,
            max_length=config.max_length,
        )
        for input_cfg in config.inputs
    ]

    merged = pd.concat(frames, ignore_index=True)
    if config.deduplicate:

        def merge_values(series: pd.Series) -> Any:
            values = [v for v in series.tolist() if pd.notna(v)]
            if not values:
                return pd.NA
            if all(v == values[0] for v in values):
                return values[0]
            unique = list(dict.fromkeys(str(v) for v in values))
            return config.merge_separator.join(unique)

        merged = merged.groupby(config.dedup_on, as_index=False).agg(merge_values)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(config.output_path, index=False)
    print(f"Saved merged file to {config.output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge multiple delimited files into a single CSV."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config describing inputs and output.",
    )
    args = parser.parse_args()

    try:
        merge_sources(Path(args.config))
    except ValidationError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
