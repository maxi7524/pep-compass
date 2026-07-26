# Locality experiments

Every experiment in this directory extends `base_locality_experiment.json`.
The shared base reads `configs/optimization/peptides/peptides_all.csv`, runs 100
SORBES trajectories per local-enumeration call, limits each candidate product
to 1,000 sequences, and uses an evaluation budget of 10. Its common grid covers
MUTANG token threshold, LAMS similarity threshold, and Levenshtein radius. The
peptide CSV is intentionally not included in this change.

Select the tracking detail in the base or an individual experiment:

- `short`: APEX evaluations only;
- `normal`: evaluations, per-step proposal counts, and provenance of evaluated
  candidates;
- `all`: every generation event after the Cartesian-product limit, with method
  and constraint-filter outcomes plus normalized parent identifiers.

The remaining independent configs cover the three-position random walker,
TANDEM top-p, TANDEM temperature excluding the top-p experiment's temperature
1.0 case, and a matched random-MUTANG baseline.

Run one experiment or the complete set from the repository root:

```bash
scripts/run_locality_experiment.sh 5 --dry-run
scripts/run_locality_experiment.sh all --execution srun --max-parallel-runs 4
```

All additional arguments are forwarded to `run_optimization.py`.
