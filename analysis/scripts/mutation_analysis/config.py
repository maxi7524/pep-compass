"""Pydantic configuration models for diff analysis."""

from pathlib import Path
from pydantic import BaseModel, Field
from typing import Literal, Optional
import yaml


class DiffConfig(BaseModel):
    """Configuration for compute_diff function."""

    value_cols: list[str] = Field(
        description="List of column names containing values to compute differences for"
    )
    value_preprocessing: Optional[Literal["log2"]] = Field(
        default=None,
        description="Preprocessing for values: None or 'log2'"
    )
    add_relative: bool = Field(description="Whether to compute relative differences")


class MutationConfig(BaseModel):
    """Configuration for mutation analysis."""

    parent_sequence_col: str = Field(description="Column name containing parent sequences")
    mutant_sequence_col: str = Field(description="Column name containing mutant sequences")
    ngram_size: int | None = Field(
        default=None,
        description="Size of n-grams for mutation analysis (if None, uses global analyses.ngram_size)"
    )
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
    seed: int = Field(description="Random seed for bootstrap sampling")


class RankConfig(BaseModel):
    """Configuration for rank computation."""

    aggregation_func: str = Field(
        description="Function to aggregate values within a bootstrap sample: 'mean' or 'sum'"
    )


class MutationFilterConfig(BaseModel):
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


class ParentDatasetFilterConfig(BaseModel):
    """Configuration for filtering the parent dataset by column values."""

    name: str = Field(
        description="Name/identifier for this filter (used in output naming)"
    )
    subdir: str | None = Field(
        default=None,
        description="Optional subdirectory to group related filters. If provided, output will be saved under {output_dir}/{group.name}/{subdir}/{name}/ instead of {output_dir}/{group.name}/{name}/"
    )
    columns: list[ColumnFilterConfig] = Field(
        description="List of column filters to apply"
    )


class CombinedPairConfig(BaseModel):
    """Configuration for a single pair within a combined group."""

    parent: Path = Field(
        description="Path to CSV file containing parent sequences and values"
    )
    mutant: Path = Field(
        description="Path to CSV file containing mutant sequences and values"
    )
    parent_csv_separator: str = Field(
        default=",",
        description="Separator/delimiter for the parent CSV file (e.g., ',', '\\t', ';'). Default is comma."
    )
    parent_column_rename_map: dict[str, str] | None = Field(
        default=None,
        description="Optional dict to rename parent file columns after loading: {old_name: new_name}. Allows multiple mappings."
    )
    mutant_column_rename_map: dict[str, str] | None = Field(
        default=None,
        description="Optional dict to rename mutant file columns after loading: {old_name: new_name}. Allows multiple mappings."
    )


class CombinedGroupConfig(BaseModel):
    """Configuration for a group of pairs to be combined."""

    name: str = Field(description="Name/identifier for this combined group")
    output_dir: Path = Field(description="Directory to save output files")
    pairs: list[list[CombinedPairConfig]] = Field(
        description="List of lists of pairs. Each inner list represents pairs that will be combined into one dataset."
    )


class FilteringConfig(BaseModel):
    """Configuration for filtering datasets."""

    parent_filters: list[ParentDatasetFilterConfig] | None = Field(
        default=None,
        description="List of parent dataset filtering criteria. If None, no parent filtering is applied. "
        "If provided, the analysis runs separately for each filter configuration.",
    )
    post_combination_filters: MutationFilterConfig = Field(
        description="Filters to apply after combining datasets (mutation filters)"
    )


class DiffOutputConfig(BaseModel):
    """Configuration for diff analysis output."""

    values: bool = Field(
        default=True, description="If True, save diff values CSV"
    )
    ranks: bool = Field(
        default=False, description="If True, compute and save ranks from diff values"
    )
    ranks_config: RankConfig | None = Field(
        default=None,
        description="Rank configuration. Required if ranks is True.",
    )


class DiffAnalysisConfig(BaseModel):
    """Configuration for diff computation analysis."""

    enabled: bool = Field(default=True, description="If False, skip diff analysis")
    config: DiffConfig = Field(description="Diff computation configuration")
    output: DiffOutputConfig = Field(description="Diff analysis output configuration")


class TransitionCountsOutputConfig(BaseModel):
    """Configuration for transition counts analysis output."""

    counts: bool = Field(
        default=True, description="If True, save aggregate mutation counter"
    )
    statistics: bool = Field(
        default=False, description="If True, compute statistics per mutation"
    )
    ranks: bool = Field(
        default=False,
        description="If True, compute and save ranks from statistics",
    )
    ranks_config: RankConfig | None = Field(
        default=None,
        description="Rank configuration. Required if ranks is True.",
    )


class TransitionCountsAnalysisConfig(BaseModel):
    """Configuration for transition counts computation analysis."""

    enabled: bool = Field(
        default=True, description="If False, skip transition counts analysis"
    )
    config: MutationConfig = Field(
        description="Mutation configuration for transition counts"
    )
    output: TransitionCountsOutputConfig = Field(
        description="Transition counts analysis output configuration"
    )


class BootstrapApplyConfig(BaseModel):
    """Configuration for which analyses to apply bootstrap to."""

    diff: bool = Field(
        default=True, description="If True, apply bootstrap to diff analysis"
    )
    transition_counts: bool = Field(
        default=True,
        description="If True, apply bootstrap to transition counts analysis",
    )


class BootstrapAnalysisConfig(BaseModel):
    """Configuration for bootstrap analysis."""

    enabled: bool = Field(
        default=False, description="If False, skip bootstrap analysis"
    )
    config: BootstrapConfig = Field(description="Bootstrap sampling configuration")
    apply_to: BootstrapApplyConfig = Field(
        description="Which analyses to apply bootstrap to"
    )


class AnalysesConfig(BaseModel):
    """Configuration for all analyses."""

    match_col_parents: str = Field(
        description="Global column name to match parents (used after remapping, if any)"
    )
    match_col_mutants: str = Field(
        description="Global column name to match mutants (used after remapping, if any)"
    )
    ngram_size: int = Field(
        description="Size of n-grams for mutation analysis (applies to all analyses, required)"
    )
    diff: DiffAnalysisConfig = Field(description="Diff analysis configuration")
    transition_counts: TransitionCountsAnalysisConfig = Field(
        description="Transition counts analysis configuration"
    )
    bootstrap: BootstrapAnalysisConfig | None = Field(
        default=None, description="Bootstrap analysis configuration (optional)"
    )


class MutationAnalysisConfig(BaseModel):
    """Configuration model for mutation difference analysis."""

    combined_groups: list[CombinedGroupConfig] = Field(
        description="List of combined dataset groups to process"
    )
    filtering: FilteringConfig = Field(description="Filtering configuration")
    analyses: AnalysesConfig = Field(description="Analyses configuration")


def load_config(config_path: Path) -> MutationAnalysisConfig:
    """Load and validate configuration from YAML file."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = MutationAnalysisConfig(**config_dict)

    # Validate that combined_groups is provided
    if not config.combined_groups:
        raise ValueError("'combined_groups' must be provided in the configuration")

    # Validate diff analysis configuration
    if config.analyses.diff.output.ranks and config.analyses.diff.output.ranks_config is None:
        raise ValueError(
            "If diff.output.ranks is True, diff.output.ranks_config must be provided"
        )

    # Validate transition_counts analysis configuration
    if (
        config.analyses.transition_counts.output.ranks
        and config.analyses.transition_counts.output.ranks_config is None
    ):
        raise ValueError(
            "If transition_counts.output.ranks is True, transition_counts.output.ranks_config must be provided"
        )
    if (
        (config.analyses.transition_counts.output.statistics or config.analyses.transition_counts.output.ranks)
        and not config.analyses.diff.enabled
    ):
        raise ValueError(
            "Diff analysis must be enabled when transition_counts.output.statistics or transition_counts.output.ranks is True. "
            "Statistics and ranks require diff value columns."
        )

    # Validate bootstrap configuration
    if config.analyses.bootstrap is not None and config.analyses.bootstrap.enabled:
        if (
            config.analyses.bootstrap.apply_to.transition_counts
            and not config.analyses.transition_counts.enabled
        ):
            raise ValueError(
                "If bootstrap.apply_to.transition_counts is True, transition_counts must be enabled"
            )
        if (
            config.analyses.bootstrap.apply_to.diff
            and not config.analyses.diff.enabled
        ):
            raise ValueError(
                "If bootstrap.apply_to.diff is True, diff must be enabled"
            )

    return config
