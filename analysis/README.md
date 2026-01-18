Run the aggregator:
`uv run python analysis/aggregate_pep_data/aggregate_pep_data.py --config analysis/aggregate_pep_data/config.yaml`

Run APEX predictions:
`uv run python analysis/apex/predict/predict.py --config analysis/apex/predict/configs/aggregate_peptides.yaml`