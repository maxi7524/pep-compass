"""Pydantic configuration models for micdiff analysis."""

from pathlib import Path
from pydantic import BaseModel, Field
from typing import Literal
import yaml


class PairConfig(BaseModel):
    """Configuration for a single parent-mutant pair."""

    parent: Path = Field(
        description="Path to CSV file containing parent sequences and values"
    )
    mutant: Path = Field(
        description="Path to CSV file containing mutant sequences and values"
    )
    output_dir: Path = Field(description="Directory to save output files")


class IOConfig(BaseModel):
    """Configuration for input/output paths (single pair)."""

    parents_dataset_path: Path = Field(
        description="Path to CSV file containing parent sequences and values"
    )
    mutants_dataset_path: Path = Field(
        description="Path to CSV file containing mutant sequences and values"
    )
    output_dir: Path = Field(description="Directory to save output files")


class DiffConfig(BaseModel):
    """Configuration for compute_diff function."""

    match_col_parents: str = Field(
        description="Column name to match parents in parents dataset"
    )
    match_col_mutants: str = Field(
        description="Column name to match parents in mutants dataset"
    )
    value_cols: list[str] = Field(
        description="List of column names containing values to compute differences for"
    )
    value_preprocessing: str | None = Field(
        description="Preprocessing for values: None or 'log'"
    )
    add_relative: bool = Field(description="Whether to compute relative differences")


class MutationConfig(BaseModel):
    """Configuration for mutation analysis."""

    parent_col: str = Field(description="Column name containing parent sequences")
    mutant_col: str = Field(description="Column name containing mutant sequences")
    ngram_size: int = Field(description="Size of n-grams for mutation analysis")
    allow_ngrams_overlap: bool = Field(
        description="Whether to allow overlapping n-grams (True) or non-overlapping (False)"
    )


class BootstrapConfig(BaseModel):
    """Configuration for bootstrap sampling."""

    n_bootstrap_samples: int = Field(
        description="Number of bootstrap samples to generate"
    )
    bootstrap_group_cols: list[str] | None = Field(
        description="Columns to group by for bootstrapping (None for rowwise)"
    )
    stratify_cols: list[str] | None = Field(
        description="Columns to stratify by for bootstrapping"
    )
    bootstrap_frac: float = Field(
        description="Fraction of data to use in bootstrap samples"
    )
    seed: int | None = Field(description="Random seed for bootstrap sampling")


class RankConfig(BaseModel):
    """Configuration for rank computation."""

    aggregation_func: str = Field(
        description="Function to aggregate values within a bootstrap sample: 'mean' or 'sum'"
    )


class FilterConfig(BaseModel):
    """Configuration for filtering mutations."""

    filter_identities: bool = Field(
        description="Filter out cases where parent sequence equals mutant sequence"
    )
    filter_length_mismatch: bool = Field(
        description="Filter out cases where parent and mutant have different lengths after stripping whitespace"
    )
    deduplicate: bool = Field(
        description="Remove duplicate rows where both parent and mutant sequences are the same"
    )


class FloatFilterConfig(BaseModel):
    """Configuration for filtering float columns."""

    dtype: Literal["float"] = "float"
    min: float | None = Field(
        default=None,
        description="Minimum value (inclusive). If None, no lower bound.",
    )
    max: float | None = Field(
        default=None,
        description="Maximum value (inclusive). If None, no upper bound.",
    )


class CategoricalFilterConfig(BaseModel):
    """Configuration for filtering categorical columns."""

    dtype: Literal["categorical"] = "categorical"
    values: list[str] = Field(description="List of categorical values to keep")


class ColumnFilterConfig(BaseModel):
    """Configuration for filtering a single column."""

    column: str = Field(description="Column name to filter")
    filter_config: FloatFilterConfig | CategoricalFilterConfig = Field(
        description="Filter configuration for this column"
    )


class ParentFilterConfig(BaseModel):
    """Configuration for a single parent filtering criteria."""

    name: str = Field(
        description="Name/identifier for this filter (used in output naming)"
    )
    columns: list[ColumnFilterConfig] = Field(
        description="List of column filters to apply"
    )


class Config(BaseModel):
    """Configuration model for mutation difference analysis."""

    # Either a single io config or a list of pairs
    io: IOConfig | None = Field(
        default=None,
        description="Single input/output configuration (deprecated, use pairs)",
    )
    pairs: list[PairConfig] | None = Field(
        default=None, description="List of parent-mutant pairs to process"
    )
    diff: DiffConfig = Field(description="Diff computation configuration")
    mutation: MutationConfig = Field(description="Mutation analysis configuration")
    bootstrap: BootstrapConfig = Field(description="Bootstrap sampling configuration")
    rank: RankConfig = Field(description="Rank computation configuration")
    filter: FilterConfig = Field(description="Filtering configuration")
    parent_filters: list[ParentFilterConfig] | None = Field(
        default=None,
        description="List of parent filtering criteria. If None, no parent filtering is applied. "
        "If provided, the analysis runs separately for each filter configuration.",
    )


class ComputeDiffConfig(BaseModel):
    """Configuration for computing diff dataframes."""

    parents_dataset_path: Path = Field(
        description="Path to CSV file containing parent sequences and values"
    )
    mutants_dataset_path: Path = Field(
        description="Path to CSV file containing mutant sequences and values"
    )
    output_path: Path = Field(description="Path to save the computed diff CSV file")
    diff: DiffConfig = Field(description="Diff computation configuration")


def load_config(config_path: Path) -> Config:
    """Load and validate configuration from YAML file."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = Config(**config_dict)

    # Validate that either io or pairs is provided
    if config.io is None and config.pairs is None:
        raise ValueError("Either 'io' or 'pairs' must be provided in the configuration")
    if config.io is not None and config.pairs is not None:
        raise ValueError("Cannot specify both 'io' and 'pairs' in the configuration")

    return config


def load_compute_diff_config(config_path: Path) -> ComputeDiffConfig:
    """Load and validate configuration from YAML file for compute_diff script."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return ComputeDiffConfig(**config_dict)
