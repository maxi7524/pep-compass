# APEX Batch Prediction

This directory contains a simple, config-driven CLI to run APEX predictions over a directory of CSV files and write outputs with the same filenames to a destination directory.

## Usage

```bash
python analysis/scripts/apex/predict/predict.py --config analysis/scripts/apex/predict/configs/veltri_negative_all.yaml
```

## Config schema

```yaml
input_dir: results/data/mutants/veltri_negative   # directory with input CSV files
output_dir: results/data/apex/veltri_negative/all # directory for outputs
file_glob: "*.csv"                               # which files to process
sequence_column: mutant                           # column with peptide sequences
use_tqdm: true                                    # show progress bar over sequences
apex_predictor_kwargs:                            # passed directly to PredictorAPEX
  path: all                                       # 'default' or 'all'
  device: cuda                                    # 'cpu' or 'cuda'
  batch_size: 1000                                # adjust to your memory
```

Notes:
- Outputs mirror input filenames and are written to `output_dir`.
- Set `sequence_column` to match your input files (default is `sequence`).
- `path` controls the model set: `default` (8 pathogens) or `all` (full set).
- Enable `use_tqdm` to see a single progress bar over the number of sequences.

## Example configs

- `analysis/scripts/apex/predict/configs/veltri_negative_all.yaml`
- `analysis/scripts/apex/predict/configs/veltri_negative_default.yaml`
- `analysis/scripts/apex/predict/configs/veltri_positive_all.yaml`
- `analysis/scripts/apex/predict/configs/veltri_positive_default.yaml`

## Note
When specifying output directories within the results for organisational purposes include the input directory as a subpath e.g.

`input_dir: results/mutants/veltri_positive`

`output_dir: results/apex_predictions/results/mutants/veltri_positive/default`