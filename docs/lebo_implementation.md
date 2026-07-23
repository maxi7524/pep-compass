# LE-BO implementation

This document explains how the historical `rl_trials` and `kjxpp/main`
experiments were decomposed into reusable components, how those components are
called, and which historical code was deliberately not copied.

## Integration design

Historical scripts repeated the APEX black box, HydrAMP, SORBES, MUTANG, local
trajectory loop, LE-BO optimizer, observer, benchmark peptides, and output
setup. Their actual experimental difference was usually one candidate scoring
or selection rule.

The integrated design gives each responsibility one location:

```text
mutation_potentials.py  score residues or complete combinations
mutation_filters.py     bound a product and select peptide strings
local_enumerator.py     run SORBES or a one-point Jacobian and enforce radius
run_optimization.py     instantiate components and execute tasks
```

This split is why an old class may not have one new class with the same name.
For example, a historical local enumerator that contained both a copied SORBES
loop and TANDEM selection becomes one shared sampling enumerator plus a TANDEM
filter.

## Historical object migration

| Historical source | Historical object or responsibility | Current object | Integration decision |
| --- | --- | --- | --- |
| `upstream/kjxpp/main:src/pep_compass/local_enumeration/mutation/mutation_potentials.py` | `DecoderLogProbPotential` | `DecoderLogProbabilityPotential` | Renamed; per-position decoder log-probability contract retained. |
| same file and its extended `upstream/rl_trials` version | `ProjectedDirectionPairwiseSimilarityPotential` | same class name | Adapted to canonical `sampling_walker.SubRiemannianTangentSpace`; original `onehot`/`diff` meaning retained. |
| same files | `compose_mutant_distribution` | same function name | Retains per-position and tuple-keyed potential composition; common candidate bound is handled before it. |
| `upstream/rl_trials:scripts/lebo_plus.py` | `_MutangPlusProductPotential` | `LamsAnchorSimilarityPotential` | Moved from script; implements the global minimum over every mutated-mutated pair. |
| same script | `DynamicSORBESMutangPlusPotential` | `_GeometryFilter._pairwise_potential()` plus `LamsAnchorSimilarityPotential` | Shared Jacobian/SVD builder replaces dynamic wrapper. |
| same script | `SamplingWithMutangPlusLocalEnumerator` | `SamplingFilteredMutationLocalEnumerator` + `LamsFilter` | Shared trajectory and LAMS policy are separated. |
| `upstream/rl_trials:scripts/lpbebo_plus.py` | `DynamicSORBESPairwiseSimilarityPotential` | `_GeometryFilter._pairwise_potential()` | Shared with LAMS and TANDEM. |
| same script | `SamplingWithMutangPlusPlusLocalEnumerator` | `SamplingFilteredMutationLocalEnumerator` + `TandemFilter` | Historical MUTANG++ label is exposed as TANDEM. |
| `upstream/rl_trials:scripts/run_lpbebo_optimization_apex.py` | one-Jacobian LPBEBO local enumerator | `FilteredMutationLocalEnumerator` + `LpbeboFilter` | Does not run a SORBES trajectory. |
| `upstream/rl_trials:scripts/move.py` | `SamplingWithMoveLocalEnumerator` | `SamplingFilteredMutationLocalEnumerator` + `MoveFilter` | First-order latent displacement policy is isolated. |
| `upstream/rl_trials:scripts/random_lebo.py` | `RandomLocalEnumerator` | `SamplingFilteredMutationLocalEnumerator` + `RandomLeBoFilter` | One filter exposes both random controls by mode. |
| `lebo_plus.py`, `lpbebo_plus.py`, `move.py`, `random_lebo.py` | repeated `_cap_mutations` | `_bounded_mutations` | One seeded candidate-product guard. |
| `lpbebo_plus.py`, `move.py` and related scripts | repeated `_top_p_filter` | `_nucleus_indices` | One descending, temperature-scaled nucleus selector. |
| `upstream/rl_trials:src/pep_compass/optimization/lebo/local_enumeration_bayesian_optimizer.py` | BLOSUM batch diversity | current LE-BO optimizer + `utils/blosum_utils.py` | Optional; original Levenshtein behavior remains the default. |

## Components not migrated

- `upstream/rl_trials:src/pep_compass/local_enumeration/sampling/sorbes.py`
  duplicates canonical `local_enumeration/sampling_walker.py` and was not copied.
- `upstream/rl_trials:src/pep_compass/local_enumeration/mutation/mutation_enumerator.py`
  duplicates canonical `local_enumeration/mutation_enumerator.py` and was not
  copied. The new `mutation/` package contains potentials and filters only.
- `AmbientMetricPairwiseSimilarityPotential`, linear pair transforms, and
  similarity-matrix helpers were used by thesis figure/analysis scripts rather
  than the optimization entry points integrated here. They remain outside the
  runtime package.
- Per-script environment parsing, benchmark dictionaries, observer setup,
  explicit CUDA cache clearing, output loops, and figure/debug code were replaced
  by JSON, CSV, task manifests, and the common runner.
- RL/A2C shell wrappers were not converted because their referenced Python
  entry points are absent on current `dev`.

## Runtime call sequence

### Task construction

The runner resolves JSON inheritance, expands `grid`, reads the starting
sequence CSV, and writes one task JSON per `(grid variant, sequence,
repetition)`. Each task has a resolved device, seed, budget, output directory,
black box, optimizer, and method. The optimizer and observer are newly
constructed for every task.

### LE-BO state

`LocalEnumerationBayesianOptimizer.optimize()` evaluates the starting point and
maintains:

- `scored_peptides`: evaluated sequence-to-score values;
- a BK-tree and set for unseen candidates;
- a fingerprint cache;
- trust-region distance and success/failure counters;
- a black-box call count constrained by `evaluation_budget`.

For every iteration, it asks the local enumerator for candidates, removes known
sequences, applies the trust region, creates MAP4 fingerprints, fits a GP with a
Tanimoto kernel, computes log Expected Improvement, selects a diverse batch,
evaluates the black box, and updates its best peptide and trust region.

### SORBES step

The sampling path starts with a latent vector of shape `[latent_dim]`.
`SubRiemannianManifold` temporarily adds the batch dimension required by
`decoder_jacobian()` and selects the first Jacobian:

```text
J: [ambient_dim, latent_dim]
U: [ambient_dim, r]
S: [r]
V (Torch Vh): [r, latent_dim]
```

For HydrAMP, `ambient_dim = 25 × 21 = 525`, `latent_dim = 64`, and `r = 64`.
SORBES splits `S` at `horizontal_threshold`, samples horizontal movement, adds
the manifold-acceleration correction, and optionally adds vertical movement.
If the proposed horizontal update exceeds `max_horizontal_update_norm`, a
Torch bisection reduces `sqrt(dt)` without moving its vectors to CPU.

The step returns the new latent vector and:

```python
{
    "adjusted_time_step": float,
    "U": torch.Tensor,
    "S": torch.Tensor,
}
```

The filtered sampling enumerator decodes the new latent vector to advance the
trajectory, but applies the neighbourhood radius to candidates relative to the
original optimizer center.

### MUTANG extraction

MUTANG selects at least `min_number_of_directions` significant singular
directions, capped by the available columns of `U`. For each direction it
reshapes its left singular vector:

```text
[max_len × alphabet_size] -> [max_len, alphabet_size]
```

It takes absolute values, excludes padding column `0`, and records every
`(position, amino_acid)` above `token_threshold`. Earlier `dev` code first
selected one position with the largest row sum. That discarded valid positions;
the current implementation checks all positions before deduplication.

Canonical `mutate()` adds the parent residue at every position and recursively
builds the full Cartesian product. Filtered methods call `_bounded_mutations()`
first because all-position extraction can enlarge that product substantially.

## Method implementations

### Baseline LE-BO

Objects:

```text
SamplingMutationLocalEnumerator
  + SecondOrderRiemannianBrownianEfficientSampling
  + MutationEnumerationInTangentSpace
LocalEnumerationBayesianOptimizer
```

There is no method-specific filter. Every complete MUTANG combination inside
the local Levenshtein radius enters the unseen candidate pool.

### LPBEBO

LPBEBO uses `FilteredMutationLocalEnumerator`, so it computes one Jacobian at
the current center and does not advance SORBES. `LpbeboFilter`:

1. bounds the MUTANG product;
2. obtains decoder log-probability for each allowed residue at the parent
   latent point;
3. adds residue log-probabilities for each complete combination;
4. sorts candidates by score;
5. divides scores by `temperature`, normalizes them, and retains the smallest
   descending prefix whose cumulative mass reaches `top_p`.

### LAMS

LAMS builds a projected-direction potential from the current parent Jacobian.
For an ambient one-hot residue direction `e`, the implementation reads its
latent pull-back from the horizontal pseudoinverse. Directions are normalized
before cosine similarity.

For every complete combination, `LamsAnchorSimilarityPotential` considers only
pairs in which both positions mutate. Its score is the minimum cosine among all
such pairs. `LamsFilter` accepts `score >= similarity_threshold`. A single
mutation has no pair, receives `+inf`, and passes any finite threshold. LAMS has
no top-p selection.

### TANDEM

TANDEM uses the same projected residue directions but includes every position
pair involving at least one mutation. For cosine `c`:

```text
both choices mutate:       log((1 + c) / 2)
exactly one choice mutates: log((1 - c) / 2)
```

The mean pair contribution is the combination log-potential. Aligned mutations
are favoured together, while choosing only one member of an aligned pair is
penalized. `TandemFilter` applies temperature-scaled top-p afterward.

`direction_mode="onehot"` represents `e_(position,target)`.
`direction_mode="diff"` represents
`e_(position,target) - e_(position,parent)` before normalization; the parent
choice therefore maps to the zero direction.

### MOVE

MOVE materializes the bounded candidate list and encodes each distinct single
substitution once. It calculates the corresponding latent displacement from the
parent. For a complete multi-mutant, it uses a first-order superposition:

```text
score(candidate) = -norm(sum(single-substitution displacements))
```

The displacement matrix and norms stay on the model device. Only the final
score vector crosses to NumPy for shared top-p selection. MOVE therefore favours
combinations predicted to remain close to the parent embedding; it does not
encode every complete multi-mutant.

### Random controls

`random_walker` ignores the real MUTANG map, samples a limited number of
positions and target residues uniformly, and materializes that product.

`random_mutang` keeps the real bounded MUTANG product, assigns random
probability mass, and keeps a random nucleus controlled by
`selection_fraction`. The first control removes geometry from proposal; the
second keeps proposal but removes geometry-aware ranking.

### BLOSUM diversity

BLOSUM is independent of local candidate potentials. After acquisition selects
the current best candidate in a batch, the optimizer either:

- removes candidates within `levenshtein_diversity_threshold`; or
- when configured, removes candidates whose BLOSUM score is not below
  `blosum_diversity_max_score`.

This affects diversity between black-box evaluations in one batch, not MUTANG
extraction or the biological objective.

## Candidate-product semantics

`_bounded_mutations()` first adds the parent residue to each mutable position.
While the Cartesian product exceeds `maximum_candidates`, it removes a random
non-parent alternative from the position with the largest choice list. If a
position contains only its parent and other positions remain, that position may
be removed. Consequently:

- candidates may mutate any subset of retained positions;
- the result is seed-dependent;
- the configured maximum limits the product before sequences are materialized;
- the parent-only combination is excluded from returned candidates.

`compose_mutant_distribution()` supports two potential contracts:

- `{position: {amino_acid: score}}`, summed across a complete combination;
- `{amino_acid_tuple: score}`, already representing a complete combination.

It returns sequences and a descending NumPy score vector with matching order.

## Device boundaries

The configured device is passed to HydrAMP and compatible black boxes. Local
enumerators derive their device from HydrAMP. SVD, tangent projection, SORBES,
pairwise potentials, MOVE displacement norms, GP fitting, and acquisition stay
on that device.

The remaining CPU/NumPy boundaries are intentional current API boundaries:

- MAP4 fingerprints are produced as NumPy-compatible data;
- POLI discrete black boxes accept NumPy arrays;
- peptide strings, mutation dictionaries, and Cartesian products are Python
  objects;
- final Torch indices/scores are converted when Python string materialization
  or NumPy top-p selection requires them.

CUDA execution could not be run in the development environment used for this
integration. Device invariants and a complete CPU SORBES step were validated;
full CUDA runtime and memory benchmarks remain required on a GPU node.

## Output traceability

Every task has a JSON file containing its resolved configuration, sequence,
repetition, seed, device, grid ID, and output path. `grid_manifest.csv` maps
parameter combinations, while `run_manifest.csv` maps concrete tasks. Observer
CSV files remain unchanged. The aggregation tool adds optimizer, black box,
method, grid ID, and source path, allowing a trajectory to be traced back to its
exact configuration.
