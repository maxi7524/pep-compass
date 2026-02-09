# Physicochemical Properties Prediction

This directory contains a simple, config-driven CLI to compute physicochemical properties for peptide sequences in a CSV file.

## Usage

```bash
python analysis/physicochemical_properties/predict.py --config analysis/physicochemical_properties/configs/aggregate_peptides.yaml
```

## Config schema

```yaml
input_file: results/data/peptides.csv                            # input CSV file
output_path: results/data/physicochemical/peptides_physicochemical.csv  # output CSV file
sequence_column: sequence                                        # column with peptide sequences
csv_separator: ","                                               # separator for input CSV
use_tqdm: true                                                   # show progress bar
ph: 7.0                                                          # pH for charge calculation
hydrophobicity_scale: Aboderin                                   # scale for hydrophobicity
properties:                                                      # which properties to compute
  - charge
  - hydrophobicity
  - isoelectric_point
  - molecular_weight
  - instability_index
  - gravy
  - aliphatic_index
  - boman_index
```

## Available properties

| Property             | Library              | Description                              |
|----------------------|----------------------|------------------------------------------|
| `charge`             | Biopython            | Net charge at the specified pH           |
| `hydrophobicity`     | peptides             | Mean hydrophobicity (configurable scale) |
| `isoelectric_point`  | Biopython            | Isoelectric point (pI)                   |
| `molecular_weight`   | Biopython            | Molecular weight in Daltons              |
| `instability_index`  | Biopython            | Instability index                        |
| `gravy`              | Biopython            | Grand average of hydropathicity          |
| `aliphatic_index`    | peptides             | Aliphatic index                          |
| `boman_index`        | peptides             | Boman (potential protein interaction) index |

## Example configs

- `analysis/physicochemical_properties/configs/aggregate_peptides.yaml`
- `analysis/physicochemical_properties/configs/aggregate_peptides_mutants.yaml`

## Note
Keep input and output paths explicit per config to avoid overwriting results.
