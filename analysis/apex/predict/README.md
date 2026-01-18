# APEX Batch Prediction

This directory contains a simple, config-driven CLI to run APEX predictions on a single CSV file.

## Usage

```bash
python analysis/apex/predict/predict.py --config analysis/apex/predict/configs/aggregate_peptides.yaml
```

## Config schema

```yaml
input_file: results/data/mutants/veltri_negative.csv  # input CSV file
output_path: results/data/apex/veltri_negative.csv    # output CSV file
sequence_column: mutant                                # column with peptide sequences
use_tqdm: true                                    # show progress bar over sequences
apex_predictor_kwargs:                            # passed directly to PredictorAPEX
  path: all                                       # 'default' or 'all'
  device: cuda                                    # 'cpu' or 'cuda'
  batch_size: 1000                                # adjust to your memory
```

Notes:
- Outputs are written to `output_path`.
- Set `sequence_column` to match your input files (default is `sequence`).
- `path` controls the model set: `default` (8 pathogens) or `all` (full set).
- Enable `use_tqdm` to see a single progress bar over the number of sequences.

## Example configs

- `analysis/apex/predict/configs/aggregate_peptides.yaml`

## Note
Keep input and output paths explicit per config to avoid overwriting results.