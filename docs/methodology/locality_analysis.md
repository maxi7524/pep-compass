# Locality experiment analysis

The analysis package indexes an experiment directory from `grid_manifest.csv`,
`run_manifest.csv`, and task JSON files. It does not load tracking tables while
building the catalog. Calls to `scan()` read only the requested CSV and process
it in bounded chunks.

```python
from pep_compass.experiments.analysis import LocalityExperiment
from pep_compass.experiments.analysis.methods import retention_summary

experiment = LocalityExperiment.open("results/locality")
selection = experiment.select(
    methods=["tandem"],
    parameters={"filter.temperature": [0.5, 2.0]},
    iteration_min=1,
)
summary = retention_summary(selection, ["method", "filter.temperature"])
```

`selection.scan("candidates")` is the memory-safe interface for custom
analyses. `collect()` is intentionally explicit because it materializes all
selected rows.

## Tracking start conditions

Detailed tracking can start at a configured optimizer iteration or when the
current center reaches a configured sequence length:

```json
{
  "tracking": {
    "level": "all",
    "store_latents": false,
    "start_iteration": 3,
    "start_sequence_length": 20
  }
}
```

The conditions use OR semantics. Tracking starts when either threshold is
reached and remains active. If both values are `null`, it starts immediately.
Late start affects tracking only; optimization still runs normally beforehand.

## Recorded filter information

For `all` tracking, every candidate after the Cartesian-product limit contains:

- `method_score`: raw LAMS, TANDEM, LPBEBO, MOVE, or random-control score;
- `method_probability`: temperature-scaled probability used by nucleus filters;
- `method_cumulative_probability`: inclusive nucleus mass in score order;
- `method_rank`: one-based descending-score rank;
- `passed_method_filter` and `passed_constraint_filter`: exact stage decisions.

Latent points do not need to be stored. `latent_distance_batches()` encodes
candidate-parent pairs offline in bounded batches using the supplied encoder.

## Count semantics

- `proposed_count`: Cartesian product size before limiting.
- `post_limit_count`: candidates remaining after `maximum_candidates`.
- `post_method_filter_count`: candidates retained by the method-specific filter.
- `post_constraint_filter_count`: candidates retained after the final
  Levenshtein constraint.

The provided notebook is `analysis/notebooks/locality_analysis.ipynb`.
