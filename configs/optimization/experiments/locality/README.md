# Locality experiments

Every experiment in this directory extends `base_locality_experiment.json`.
The shared base reads `configs/optimization/peptides/peptides_all.csv`, runs 100
SORBES trajectories per local-enumeration call, and uses an evaluation budget
of 10 so an `all` tracking run normally finishes within a few BO iterations.
The peptide CSV is intentionally not included in this change.

Select the tracking detail in the base or an individual experiment:

- `short`: APEX evaluations only;
- `normal`: evaluations, per-step proposal counts, and provenance of evaluated
  candidates;
- `all`: every generation event after the Cartesian-product limit, with method
  and constraint-filter outcomes plus normalized parent identifiers.

The numbered configs cover MUTANG token sensitivity, LAMS thresholds and
Levenshtein radii, TANDEM top-p and temperature, and random controls.
