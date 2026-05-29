# Mutation Analysis

Tool for computing bootstrap statistics for mutation analysis, comparing parent and mutant sequences using n-gram mutation detection and statistical ranking.

## Running the Application

```bash
# Process a single config file
uv run python -m analysis.scripts.mutation_analysis.mutation_analysis_app --config path/to/config.yaml

# Process all YAML files in a directory
uv run python -m analysis.scripts.mutation_analysis.mutation_analysis_app --config path/to/configs/
```

**Arguments:**
- `--config`: Path to YAML configuration file or directory containing YAML files (required)
  - If a file is provided, processes that single file
  - If a directory is provided, processes all `.yaml` and `.yml` files in the directory

**Features:**
- **Automatic output checking**: The tool automatically checks if output files already exist before processing. If all expected outputs exist for a filter/dataset combination, it skips the computation entirely, saving time on re-runs.
- **Early exit**: Output checking happens before loading data, so expensive data loading is skipped when outputs already exist.

**Note:** The seed for bootstrap sampling is configured in the YAML file under `analyses.bootstrap.config.seed`.

## Configuration Files

### Main Application Config

Configs are in `configs/mutation_analysis_app/`. Create a YAML file with this structure:

```yaml
# Combined dataset groups - list of lists of pairs to combine
# Each inner list represents pairs that will be combined into one dataset
combined_groups:
  - name: "group_1"
    output_dir: path/to/output/group_1
    pairs:  # First combined dataset (combines these pairs)
      - parent: path/to/parent1.csv
        mutant: path/to/mutant1.csv
        match_col_parents: "Name"
        match_col_mutants: "parent_name"
      - parent: path/to/parent2.csv
        mutant: path/to/mutant2.csv
        match_col_parents: "Name"
        match_col_mutants: "parent_name"
  - name: "group_2"
    output_dir: path/to/output/group_2
    pairs:  # Second combined dataset
      - parent: path/to/parent3.csv
        mutant: path/to/mutant3.csv
        match_col_parents: "Name"
        match_col_mutants: "parent_name"

# Filtering configuration - applies to all combined groups
filtering:
  # Parent filters - applied before combination, each creates a separate result directory
  parent_filters:
    - name: filter_name_1
      subdir: optional_subdir  # Optional: groups related filters together
      columns:
        - column: activity
          filter_config:
            dtype: float
            min: 10.0
            max: 100.0
    - name: filter_name_2
      subdir: optional_subdir  # Filters with same subdir are grouped together
      columns:
        - column: category
          filter_config:
            dtype: categorical
            values: ["type_A", "type_B"]
  # Post-combination filters - applied after combining datasets
  post_combination_filters:
    filter_identities: true
    filter_length_mismatch: true
    deduplicate: true

# Analyses configuration - specify which analyses to perform
analyses:
  # Diff computation analysis
  diff:
    enabled: true
    config:
      match_col_parents: Name
      match_col_mutants: parent_name
      value_cols: ["value1", "value2"]
      value_preprocessing: null          # null or "log"
      add_relative: true
    output:
      values: true                        # Save diff values CSV
      ranks: false                        # Compute and save ranks from diff values
      ranks_config:                       # Only used if ranks: true
        aggregation_func: mean            # "mean" or "sum"

  # Transition counts (mutation) analysis
  transition_counts:
    enabled: true
    config:
      parent_col: parent
      mutant_col: mutant
      ngram_size: 3
      allow_ngrams_overlap: true
    output:
      counts: true                        # Save aggregate mutation counter
      statistics: false                   # Compute statistics per mutation (requires diff enabled)
      ranks: false                        # Compute and save ranks from statistics (requires diff enabled)
      ranks_config:                       # Only used if ranks: true
        aggregation_func: mean            # "mean" or "sum"

  # Bootstrap configuration (optional)
  bootstrap:
    enabled: true
    config:
      n_bootstrap_samples: 100
      bootstrap_group_cols: null         # null for rowwise
      stratify_cols: null
      bootstrap_frac: 1.0
      seed: 42
    # Which analyses to apply bootstrap to
    apply_to:
      diff: true
      transition_counts: true
```

See `configs/mutation_analysis_app/` for example configurations.

## Output

For each combined group, the tool generates output files organized as follows:

```
{group.output_dir}/
  {group.name}/                       # Group name from config
    {parent_filter.subdir}/           # If parent_filter.subdir is specified
      {parent_filter.name}/           # If parent_filters specified, else skip filter levels
        diff_values.csv               # If diff.output.values is True
        diff_ranks.pkl                # If diff.output.ranks is True
        transition_counts.pkl         # If transition_counts.output.counts is True
        transition_counts_statistics.pkl  # If transition_counts.output.statistics is True
        transition_counts_ranks.pkl   # If transition_counts.output.ranks is True
        bootstrap/                    # If bootstrap.enabled is True
          diff/                       # If bootstrap.apply_to.diff is True
            seed{seed}_n{n_samples}/
              1.pkl, 2.pkl, ..., {n_bootstrap_samples}.pkl
          transition_counts/          # If bootstrap.apply_to.transition_counts is True
            seed{seed}_n{n_samples}/
              1.pkl, 2.pkl, ..., {n_bootstrap_samples}.pkl
```

**Output directory structure:**
- If `parent_filter.subdir` is specified: `{group.output_dir}/{group.name}/{subdir}/{filter_name}/`
- If only `parent_filter.name` is specified: `{group.output_dir}/{group.name}/{filter_name}/`
- If no parent filters: `{group.output_dir}/{group.name}/`

**Output files:**
- `diff_values.csv`: DataFrame with computed differences between parents and mutants (if `diff.output.values` is True)
- `diff_ranks.pkl`: Pickled dictionary with ranks computed from diff values (if `diff.output.ranks` is True)
- `transition_counts.pkl`: Pickled aggregate mutation counter dictionary (if `transition_counts.output.counts` is True)
- `transition_counts_statistics.pkl`: Pickled statistics per mutation (if `transition_counts.output.statistics` is True)
- `transition_counts_ranks.pkl`: Pickled ranks computed from transition counts statistics (if `transition_counts.output.ranks` is True)
- Bootstrap sample files: Individual rank dictionaries for each bootstrap sample in `bootstrap/{analysis_type}/seed{seed}_n{n_samples}/` (if bootstrap is enabled)

**Note**: 
- Each analysis type (diff and transition_counts) can be enabled/disabled independently. 
- Bootstrap can be applied to either or both analysis types separately.
- **Transition statistics and ranks require diff analysis to be enabled** (they need diff value columns to compute statistics per mutation).

## Processing Flow

The application processes all combinations of combined groups × parent filters × datasets:

For each combined group:
1. **For each list of pairs in the group**:
   - **Check outputs for all filters**: Before loading any data, check if all expected output files already exist for each parent filter
   - **Skip if complete**: If all outputs exist for a filter, skip processing entirely (saves time on re-runs)
   - **Load data only if needed**: Only load and combine pairs if at least one filter needs processing
   - **Combine pairs**: Load all parent and mutant files, match each pair, and concatenate into single combined datasets
   - **For each parent filter that needs processing** (or no filter):
     - **Apply parent filtering** (if configured): Filters both datasets based on parent filter criteria
     - **Compute diff DataFrame**: Calculates differences between matched parent-mutant pairs
     - **Apply post-combination filters**: Filters identities, length mismatches, and deduplicates on the diff DataFrame
     - **Run diff analysis** (if enabled):
       - Save diff values CSV (if `diff.output.values` is True)
       - Compute and save ranks from diff values (if `diff.output.ranks` is True)
     - **Run transition counts analysis** (if enabled):
       - Compute aggregate mutation counter
       - Save counts (if `transition_counts.output.counts` is True)
       - Compute and save statistics per mutation (if `transition_counts.output.statistics` is True)
         - **Requires diff analysis to be enabled** (uses diff value columns)
         - Statistics are computed by aggregating diff values per mutation across all occurrences
       - Compute and save ranks from statistics (if `transition_counts.output.ranks` is True)
         - **Requires diff analysis to be enabled** (uses diff value columns)
         - Ranks are computed from aggregated statistics per mutation
     - **Run bootstrap analyses** (if enabled):
       - Bootstrap diff analysis (if `bootstrap.apply_to.diff` is True)
       - Bootstrap transition counts analysis (if `bootstrap.apply_to.transition_counts` is True)

## Notes

- All paths are relative to project root
- Output directories are created automatically
- **Output checking**: The tool automatically checks if all expected outputs exist before processing. This means:
  - Re-running the same config will skip already-computed results
  - Output checking happens before data loading, saving time
  - All expected files must exist (diff values, ranks, transition counts, bootstrap samples, etc.) for a filter to be skipped
- Parent filters run the analysis separately for each filter configuration, creating subdirectories
- Output directory structure: `{group.output_dir}/{group.name}/{subdir}/{filter_name}/` (subdir is optional)
- Diff and transition_counts analyses can be enabled/disabled independently
- **Transition statistics and ranks require diff analysis to be enabled** (they need diff value columns)
- Bootstrap can be applied independently to diff analysis, transition_counts analysis, or both
- Each analysis type has its own rank configuration (ranks_config) with aggregation function
- Bootstrap sample files are numbered starting from 1 (no leading zeros)
- Bootstrap output directory format: `bootstrap/{analysis_type}/seed{seed}_n{n_samples}/`
- When combining pairs, all pairs in a list are matched and concatenated before filtering and analysis
- Transition statistics compute aggregated values (mean/sum) per mutation from diff values across all occurrences
- **Diff ranks**: Diff ranks are computed by aggregating diff values by mutations and then ranking the aggregated values (similar to transition_counts ranks)
