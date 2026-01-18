Run the aggregator (`aggregate_pep_dat`):
`uv run python analysis/aggregate_pep_data/aggregate_pep_data.py --config analysis/aggregate_pep_data/config.yaml`

Run APEX predictions (`apes/predict`):
`uv run python analysis/apex/predict/predict.py --config analysis/apex/predict/configs/aggregate_peptides.yaml`

Generate mutations from the aggregated peptides (`generate_mutations`):
`uv python analysis/generate_mutations/enumerate_mutations.py --config analysis/generate_mutations/configs/aggregate_peptides.yaml`