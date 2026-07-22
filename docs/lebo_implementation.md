# LE-BO integration internals

This document explains how the standalone experiments from `rl_trials` and
`kjxpp/main` were decomposed and integrated into the current package. It is a
code-level companion to `optimization_runners.md`, which covers execution and
configuration.

## Why the standalone scripts were not copied

Each historical script constructed the same APEX black box, HydrAMP model,
SORBES walker, MUTANG enumerator, LE-BO optimizer, observer, peptide benchmark,
and run loop. It then embedded one method-specific potential and a slightly
modified copy of the local-enumeration loop. Copying the scripts would preserve
several independent versions of the same algorithm and make later fixes depend
on which entry point was used.

The integrated structure separates four responsibilities:

1. `mutation_potentials.py` assigns scores to residues or complete mutation
   combinations.
2. `mutation_filters.py` limits the Cartesian product and selects peptide
   candidates using a potential or random control.
3. `local_enumerator.py` owns either the SORBES trajectory or a one-point MUTANG
   enumeration and delegates candidate selection to a filter.
4. `run_lebo_apex.py` constructs those objects from a resolved configuration and
   runs the common benchmark loop.

## Object migration map

| Historical branch and object | Historical file | Integrated location | Change |
| --- | --- | --- | --- |
| `ProjectedDirectionPairwiseSimilarityPotential` | `rl_trials:src/pep_compass/local_enumeration/mutation/mutation_potentials.py` | `mutation_potentials.ProjectedDirectionPairwiseSimilarityPotential` | Retained and vectorized; uses canonical `sampling_walker.SubRiemannianTangentSpace`. |
| `DecoderLogProbPotential` | `kjxpp/main:src/pep_compass/local_enumeration/mutation/mutation_potentials.py` | `mutation_potentials.DecoderLogProbabilityPotential` | Renamed for clarity; return format is unchanged. |
| `_MutangPlusProductPotential` | `rl_trials:scripts/lebo_plus.py` | `mutation_potentials.LamsAnchorSimilarityPotential` | Moved out of the script; despite the retained class name, it implements the original global minimum pairwise score. |
| `DynamicSORBESMutangPlusPotential` | `rl_trials:scripts/lebo_plus.py` | `_GeometryFilter._pairwise_potential` plus `LamsAnchorSimilarityPotential` | Dynamic Jacobian/SVD construction is shared with TANDEM. |
| `DynamicSORBESPairwiseSimilarityPotential` | `rl_trials:scripts/lpbebo_plus.py` and `run_lpbebo_optimization_apex.py` | `_GeometryFilter._pairwise_potential` | One shared tangent-space builder replaces both copies. |
| `SamplingWithMutangPlusLocalEnumerator` | `rl_trials:scripts/lebo_plus.py` | `SamplingFilteredMutationLocalEnumerator` plus `LamsFilter` | SORBES loop and LAMS selection are separated. |
| `SamplingWithMutangPlusPlusLocalEnumerator` | `rl_trials:scripts/lpbebo_plus.py` | `SamplingFilteredMutationLocalEnumerator` plus `TandemFilter` | The old MUTANG++ name is represented as TANDEM. |
| `SamplingWithMoveLocalEnumerator` | `rl_trials:scripts/move.py` | `SamplingFilteredMutationLocalEnumerator` plus `MoveFilter` | Its first-order displacement scoring is isolated in the filter. |
| `RandomLocalEnumerator` | `rl_trials:scripts/random_lebo.py` | `SamplingFilteredMutationLocalEnumerator` plus `RandomLeBoFilter` | Both random controls are selected by `mode`. |
| Per-script `_top_p_filter` | `lpbebo_plus.py` and `move.py` | `mutation_filters._nucleus_indices` | One descending, temperature-scaled nucleus selector. |
| Per-script `_cap_mutations` | `lebo_plus.py`, `lpbebo_plus.py`, `move.py`, `random_lebo.py` | `mutation_filters._bounded_mutations` | One budget guard retaining the parent choice at each remaining position. |
| Per-script APEX benchmark and environment variables | all five scripts | `run_lebo_apex.py`, `configs/lebo_apex/*.json`, and `peptides.csv` | Object construction, parameters, and input sequences are data-driven. |

`LamsAnchorSimilarityPotential` retains its current public name to avoid another
rename in the same integration, but its implementation now follows the
`_MutangPlusProductPotential` author intent exactly: the minimum over all
mutated-mutated pairs, not a best-anchor maximum.

## Components deliberately not copied

- `rl_trials:src/pep_compass/local_enumeration/sampling/sorbes.py` duplicated
  `src/pep_compass/local_enumeration/sampling_walker.py`. The latter remains the
  only SORBES implementation.
- `rl_trials:src/pep_compass/local_enumeration/mutation/mutation_enumerator.py`
  duplicated the canonical `mutation_enumerator.py`. Only potentials and filters
  use the new `mutation` subpackage.
- `AmbientMetricPairwiseSimilarityPotential`, linear pair transforms,
  `compute_similarity_matrix`, and figure-specific helpers were used by thesis
  analysis scripts under `analysis/scripts/thesis_figures`. They were not used
  by the APEX optimization entry points integrated here and were therefore not
  moved into the runtime package.
- Environment parsing, hard-coded benchmark dictionaries, top-level execution,
  repeated APEX/HydrAMP construction, and repeated `CSVObserver` setup were
  replaced by the runner and JSON/CSV configuration.
- Explicit CUDA cache clearing and the separate CPU script were not copied.
  Device behavior is configured by `device`; broader CPU/CUDA optimization was
  intentionally deferred.
- Historical debug prints, per-step retention counters, and temporary table or
  figure generation were not part of the optimization algorithm and were not
  moved.

## End-to-end optimization flow

For each grid variant, input sequence, and repetition, the runner performs the
following process.

### 1. Runner construction

`run_lebo_apex.py` resolves inherited JSON, expands the Cartesian product in
`grid`, and loads `name`, `sequence`, and `repetitions` from the input CSV. It
constructs one APEX black box, observer, HydrAMP encoder-decoder, local
enumerator, and `LocalEnumerationBayesianOptimizer` for the variant.

The input sequence is not training data for HydrAMP. It is the starting point
passed to `optimizer.optimize`. Every repetition starts a new optimization from
that peptide with a different seed.

### 2. LE-BO outer loop

The optimizer evaluates the starting peptide, asks the local enumerator for a
discrete candidate neighbourhood, and stores unseen candidates in a BK-tree. It
represents peptides with MAP4 fingerprints, fits a `SingleTaskGP` using a
Tanimoto kernel, and ranks candidates with log Expected Improvement. The trust
region filters candidates by Levenshtein distance from the current best peptide.

Selected acquisition candidates are evaluated by APEX. Within a batch, the
optimizer enforces either Levenshtein diversity or the optional BLOSUM score
threshold. The APEX observer writes every evaluation to a trajectory CSV.

### 3. SORBES trajectory

`SamplingFilteredMutationLocalEnumerator` starts at the HydrAMP embedding of the
optimizer center. At each step, canonical SORBES:

1. computes the decoder Jacobian and its SVD;
2. separates horizontal and vertical directions using `horizontal_threshold`;
3. samples a horizontal Brownian direction and optional vertical movement;
4. applies the second-order correction and bounds the update norm;
5. returns the new latent point, adjusted time step, and the local `U` and `S`.

The decoded peptide advances the trajectory. Candidate peptides generated at
each step are still filtered by Levenshtein distance to the original center of
that local enumeration call.

LPBEBO is the exception: `FilteredMutationLocalEnumerator` computes one
Jacobian/SVD at the center without running a SORBES trajectory.

### 4. MUTANG extraction

For every selected singular direction, `get_mutations_from_s_u` reshapes the
corresponding left singular vector from
`max_len * alphabet_size` to `(max_len, alphabet_size)`. It takes the absolute
weight and records every non-padding `(position, residue)` pair meeting
`token_threshold`.

The historical `dev` implementation selected only the position with the largest
row sum for each direction. The integrated implementation checks all positions,
matching the nested position/residue thresholding in Algorithm 5. Duplicate
residue choices are removed before candidate composition.

### 5. Candidate-product bound

MUTANG can expose a large product of positions and residues. Before materializing
peptides, `_bounded_mutations` adds the parent residue at each position and
randomly removes non-parent alternatives from the largest choice list until the
product is at most `maximum_candidates`. If necessary, an entire position with
no removable alternative can be dropped. This bound is stochastic and therefore
depends on the run seed.

### 6. Method-specific selection

#### Baseline LE-BO

`SamplingMutationLocalEnumerator` uses canonical MUTANG `mutate`, which builds
the full allowed product and applies only the Levenshtein radius. It has no
additional candidate filter.

#### LPBEBO

`DecoderLogProbabilityPotential` encodes the parent, evaluates decoder
log-probabilities at that latent point, and reads the score for each MUTANG
residue. Scores are summed across positions for every complete candidate. After
temperature scaling and softmax, the smallest descending prefix whose
cumulative mass reaches `top_p` is retained.

#### LAMS

The decoder Jacobian at the current parent is decomposed. Ambient residue
one-hot directions are pulled back with the pseudoinverse of the horizontal
Jacobian and normalized. Parent residues are included so candidates may mutate
only a subset of positions.

For a multi-mutant, LAMS computes the cosine for every pair of mutations and
uses the minimum pairwise cosine as its score. The hard rule
`score >= similarity_threshold` keeps only candidates for which all mutation
pairs are sufficiently aligned. A single mutation has no pair, receives
`+inf`, and always passes the finite threshold. There is no top-p step.

#### TANDEM

TANDEM uses the same projected directions but scores every involved position
pair. For cosine `c`:

```text
both residues mutated:       log((1 + c) / 2)
exactly one residue mutated: log((1 - c) / 2)
```

The mean contribution is the candidate log-potential. Aligned mutations are
rewarded when taken together, while taking only one member of an aligned pair is
penalized. Temperature-scaled top-p selection is then applied.

#### MOVE

MOVE enumerates candidates, encodes every distinct single substitution once,
and caches its latent displacement from the parent. For a multi-mutant it adds
those single-mutant displacements as a first-order approximation and assigns
`-norm(sum(displacements))`. Top-p selection therefore favours combinations
predicted to stay close to the local parent embedding.

#### Random controls

`random_walker` discards the MUTANG residue map and samples positions and target
residues uniformly before building the candidate product. `random_mutang` keeps
the real MUTANG product but assigns random probability mass and retains a random
nucleus controlled by `selection_fraction`. These controls isolate geometry-
aware proposal and filtering effects.

## Output identity

`grid_manifest.csv` records each parameter combination. Every grid directory
contains the fully resolved configuration and one observer CSV per optimization
run. The aggregation script adds `grid_id`, method, and source path without
changing the original trajectory. Consequently a result can be traced back to
the exact method, parameters, starting sequence, repetition seed, and raw APEX
evaluations.
