# Mutation Difference Analysis (micdiff)

Tool for computing bootstrap statistics for mutation analysis, comparing parent and mutant sequences using n-gram mutation detection and statistical ranking.

## Running the Applications

### Main Application

```bash
uv run python -m analysis.scripts.micdiff.micdiff_app --config path/to/config.yaml
```

### Standalone Diff Computation

```bash
uv run python -m analysis.scripts.micdiff.compute_diff --config path/to/compute_diff_config.yaml
```

## Configuration Files

### Main Application Config

Configs are in `configs/micdiff_app/`. Create a YAML file with this structure:

```yaml
pairs:
  - parent: path/to/parents.csv          # Relative to project root
    mutant: path/to/mutants.csv
    output_dir: path/to/output/

diff:
  match_col_parents: Name
  match_col_mutants: parent_name
  value_cols: ["value1", "value2"]
  value_preprocessing: null              # null or "log"
  add_relative: true

mutation:
  parent_col: parent
  mutant_col: mutant
  ngram_size: 3
  allow_ngrams_overlap: true

bootstrap:
  n_bootstrap_samples: 100
  bootstrap_group_cols: null             # null for rowwise
  stratify_cols: null
  bootstrap_frac: 1.0
  seed: 42

rank:
  aggregation_func: mean                 # "mean" or "sum"

filter:
  filter_identities: true
  filter_length_mismatch: true
  deduplicate: true

# Optional: Parent filtering
parent_filters:
  - name: filter_name
    columns:
      - column: activity
        filter_config:
          dtype: float                   # "float" or "categorical"
          min: 10.0                      # For float: min/max
          max: 100.0
      - column: category
        filter_config:
          dtype: categorical
          values: ["type_A", "type_B"]   # For categorical: list of values
```

See `configs/micdiff_app/veltri.yaml` for a complete example.

### Compute Diff Config

Configs are in `configs/compute_diff/`. Create a YAML file with this structure:

```yaml
parents_dataset_path: path/to/parents.csv
mutants_dataset_path: path/to/mutants.csv
output_path: path/to/output/diff.csv

diff:
  match_col_parents: Name
  match_col_mutants: parent_name
  value_cols: ["value1", "value2"]
  value_preprocessing: null              # null or "log2"
  add_relative: true
```

## Output

For each pair, the tool generates:

- `{mutant_name}_diff.csv`: DataFrame with computed differences
- `{mutant_name}_bootstrap_sample_ranks.pkl`: Pickled dictionary with ranks per mutation per bootstrap sample

When using `parent_filters`, each filter creates a subdirectory in the output directory.

## Notes

- All paths are relative to project root
- Output directories are created automatically
- Parent filters run the analysis separately for each filter configuration
