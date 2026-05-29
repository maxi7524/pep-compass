# Mutation Enumeration Script

## Overview

The `enumerate_mutations_single_position.py` script processes peptide sequences and generates single-position mutations using tangent-space enumeration with the HydrAMP encoder-decoder model.

## Key Features

- **Command-line interface** with configurable parameters
- **Automatic device detection** (CUDA/CPU)
- **Progress tracking** with tqdm
- **Error handling** and file validation
- **Flexible input/output** paths
- **Documented functions** with type hints

## Usage

### Configuration File

The script uses a YAML configuration file to specify all parameters. The `--config` argument accepts either:

1. **A single YAML configuration file** - processes one configuration
2. **A directory path** - processes all YAML files (`.yaml` or `.yml`) in the directory sequentially

**Single config file:**
```bash
python analysis/scripts/mutang/enumerate_mutations_single_position.py --config config.yaml
```

**Directory of config files:**
```bash
python analysis/scripts/mutang/enumerate_mutations_single_position.py --config configs/
```

When a directory is provided, the script will:
- Find all `.yaml` and `.yml` files in the directory
- Process them sequentially in alphabetical order
- Display progress for each config file
- Generate separate output files for each configuration

### Example Configuration

Create a `config.yaml` file with the following structure:

```yaml
dataset_path: results/data/input_peptides.csv
output_dir: results/data/mutants
weights_dir: results/weights/hydramp

seq_col: Sequence
direction_threshold: 0.0001
token_threshold: 0.05
jacobian_mode: approx

parent_keep_columns_rename_map:
  Name: parent_name
  is_positive: is_positive
```

See `config.example.yaml` for a complete example.

## Command-line Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--config` | str | **Yes** | Path to YAML configuration file or directory containing YAML config files. If a directory is provided, all `.yaml` and `.yml` files in that directory will be processed sequentially. |

### Configuration Parameters

All other parameters are specified in the YAML config file:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_path` | Path | - | Path to input CSV file containing peptide sequences |
| `output_dir` | Path | - | Directory to save output CSV files (filename based on input name + parameters) |
| `weights_dir` | Path | - | Directory containing encoder and decoder weights |
| `seq_col` | str | `Sequence` | Name of the column containing sequences |
| `direction_threshold` | float or list | `0.0001` | Direction significance threshold(s) |
| `token_threshold` | float or list | `0.05` | Token threshold(s) for mutations |
| `jacobian_mode` | str | `approx` | Jacobian mode (`strict` or `approx`) |
| `parent_keep_columns_rename_map` | dict | `{}` | Map parent columns to output column names |

### Parameter Sweeps

You can specify lists of values for `direction_threshold` and `token_threshold` to run parameter sweeps:

```yaml
direction_threshold: [0.0001, 0.001, 0.01]
token_threshold: [0.05, 0.1, 0.2]
```

**Important:** Both lists must have the same length. Parameters are paired by index (not all combinations):

For input `veltri_positive.csv`, outputs will be:
- `veltri_positive_direction_threshold=0.0001_token_threshold=0.05_jacobian_mode=approx.csv`
- `veltri_positive_direction_threshold=0.001_token_threshold=0.1_jacobian_mode=approx.csv`
- `veltri_positive_direction_threshold=0.01_token_threshold=0.2_jacobian_mode=approx.csv`

## Output Format

The output CSV file contains the following columns:

- `mutant`: The mutated peptide sequence
- `position`: Position where the mutation occurred
- `parent`: The original peptide sequence
- `direction_significance_threshold`: Threshold used for direction significance
- `token_threshold`: Threshold used for token mutations
- Additional columns from the parent dataset (as specified in `parent_keep_columns_rename_map`)

## How It Works

1. **Model Loading**: Loads the HydrAMP encoder-decoder model with pre-trained weights
2. **Peptide Encoding**: Encodes each peptide sequence into latent space
3. **Jacobian Computation**: Computes the jacobian matrix
4. **SVD Decomposition**: Performs singular value decomposition
5. **Mutation Enumeration**: Identifies significant directions for mutations
6. **Mutant Generation**: Generates single-position mutations based on thresholds
7. **Result Export**: Saves results to CSV with metadata

## Example Output

```
Using device: cuda
Model loaded successfully
Project root: /path/to/pep-compass
Loading positive examples from: results/data/veltri_positive.csv
Loading negative examples from: results/data/veltri_negative.csv
Total sequences to process: 100

Processing peptides: 100%|████████████| 100/100 [00:05<00:00, 18.23it/s]

Total mutants generated: 1542
Unique mutants: 1240

Saving results to: results/data/mutants/veltri_mutants_mode=approx_dthresh=0.0001_tthresh=0.05.csv
Done!
```

## Requirements

- PyTorch
- pandas
- tqdm
- NumPy
- pep_compass package

## Troubleshooting

### CUDA Out of Memory

If you encounter CUDA out of memory errors, try:

```bash
CUDA_VISIBLE_DEVICES="" python results/scripts/mutang/enumerate_mutations_single_position.py
```

This forces CPU usage.

### File Not Found Errors

Ensure the input CSV files exist and paths are correct:

```bash
python results/scripts/mutang/enumerate_mutations_single_position.py \
    --input-positive /full/path/to/positive.csv \
    --input-negative /full/path/to/negative.csv
```

### Model Weights Not Found

Ensure the model weights are in the expected location:
- `results/weights/hydramp/encoder_weights.pickle`
- `results/weights/hydramp/decoder_weights.pickle`

