# Library architecture

## Workflow

PepCompass separates peptide optimization into four runtime layers:

1. a model maps peptide strings to latent tensors, maps latent tensors back to
   strings, and exposes decoder derivatives;
2. local enumeration creates a finite set of peptide candidates near a current
   center;
3. an optimizer chooses candidates and controls the evaluation budget;
4. a black box scores candidates and an observer records the trajectory.

The configured experiment path is:

```text
JSON configuration + peptide CSV
  -> scripts/runner/run_optimization.py
  -> black-box factory + optimizer factory
  -> optional HydrAMP encoder-decoder
  -> optional SORBES -> MUTANG -> candidate filter
  -> optimizer.optimize(starting_point, evaluation_budget, seed)
  -> black box model
  -> CSVObserver
  -> trajectory CSV + grid/run manifests
```

The input CSV is not model training data. Each `sequence` is an independent
optimization starting point. `repetitions` creates additional runs with distinct
seeds. The JSON controls object construction and can expand selected values into
a Cartesian parameter grid. See [Optimization runner](optimization_runners.md)
for its complete schema.

### LE-BO request flow

For LE-BO, one optimization iteration follows this path:

```text
current best peptide
  -> LocalEnumerator.local_enumeration()
  -> unseen peptide set / BK-tree
  -> MAP4 fingerprints
  -> SingleTaskGP with Tanimoto kernel
  -> log Expected Improvement
  -> diversity filter
  -> black-box evaluation
  -> optimizer state and trust-region update
```

The local enumerator may use a latent trajectory:

```text
peptide
  -> HydrAMPEncoderDecoder.encode_peptides()       [batch, latent_dim]
  -> decoder Jacobian                              [batch, ambient_dim, latent_dim]
  -> SVD: U [ambient_dim, r], S [r], Vh [r, latent_dim]
  -> SORBES latent step
  -> MUTANG map {position: [amino-acid indices]}
  -> optional LPBEBO/LAMS/TANDEM/MOVE/random filter
  -> peptide candidates within the local radius
```

All geometry tensors remain on the encoder-decoder device. Conversion to NumPy
is limited to external APIs that require it, such as MAP4/POLI calls, and to
small final index or score arrays used to materialize Python strings.

## Code structure

| Path | Responsibility |
| --- | --- |
| `src/pep_compass/models/` | neural models, predictors, and the encoder-decoder abstraction |
| `src/pep_compass/local_enumeration/` | latent walkers, MUTANG, candidate potentials, filters, and neighbourhood orchestration |
| `src/pep_compass/optimization/` | optimizer contracts, LE-BO, baselines, black-box adapters, and observers |
| `src/pep_compass/utils/` | sequence, seed, timing, and BLOSUM helpers |
| `scripts/` | executable composition, result aggregation, model initialization, and legacy entry points |
| `configs/optimization/` | complete runner defaults, experiment presets, grid templates, and peptide CSV |
| `analysis/` | downstream analysis and exploratory material; it is not runtime library code |
| `docs/` | architecture, algorithm, and runner documentation |

Implementation belongs under `src/pep_compass`. A new experiment using existing
components normally belongs in a JSON file, not in another copied Python runner.

## Models module

### Encoder-decoder

`models/encoder_decoder/encoder_decoder.py` defines the derivative operations
required by geometry. Its central shape contract is:

- decoder input: `(batch, latent_dim)`;
- decoder output: `(batch, ambient_dim)` when flattened;
- decoder Jacobian: `(batch, ambient_dim, latent_dim)`.

`HydrAMPEncoderDecoder` implements that contract. It provides:

- `encode_peptides()` and `encode_peptides_with_std()`;
- `decode_peptides()`;
- exact or finite-difference decoder Jacobians;
- finite-difference field derivatives used by SORBES.

HydrAMP uses a 64-dimensional latent space and a flattened ambient space of
`25 × 21 = 525`. The 21 tokens include padding. Its encoder and decoder weights
are stored under `models/hydramp/weights`.

### Predictive models

Model-specific subpackages provide inference implementations:

- `models/apex/`: APEX pathogen MIC models;
- `models/battleamp/`: BattleAMP predictor;
- `models/toxipep/`: ToxiPep predictor;
- `models/hydrophobicity/`: deterministic scale-based predictor;
- `models/esm/`: ESM2 pseudo-log-likelihood scoring;
- `models/EIPred/` and `models/mbc_attention/`: additional predictors not
  currently registered in the common runner.

APEX weights are external. Install them with
`scripts/initialization/download_apex_models.sh`. Other model directories mix
PyTorch, TensorFlow/Keras, and stored assets; importing or running every model
may require different optional environments.

## Local enumeration module

### Sampling walker

`local_enumeration/sampling_walker.py` contains the canonical SORBES code:

- `SubRiemannianManifold` requests the decoder Jacobian and computes its SVD;
- `SubRiemannianTangentSpace` partitions directions by singular-value threshold,
  samples horizontal/vertical directions, and caches a horizontal projection;
- `SecondOrderRiemannianBrownianEfficientSampling` computes a second-order step,
  reduces its time step when the update exceeds the norm bound, and returns the
  new latent point with `U`, `S`, and `adjusted_time_step`.

The historical `local_enumeration/sampling/sorbes.py` is not part of the current
package. `sampling_walker.py` is the only SORBES implementation.

`SORBESWithoutManifoldAcceleration` is currently unused and its constructor does
not match the active base-class API; a source TODO marks it as unavailable until
the intended historical contract is established.

### Mutation enumerator

`local_enumeration/mutation_enumerator.py` contains canonical MUTANG:

- `MutationEnumerationInTangentSpace.get_mutations_from_s_u()` converts `U` and
  `S` into allowed residue indices for every position above the token threshold;
- `mutate_peptide()` constructs complete peptide combinations;
- `AblationRandomMutationEnumerator` is the random mutation-map ablation.

The method accepts NumPy arrays or Torch tensors. Torch inputs stay on their
device while thresholding; only the final small index table moves to Python.
The number of requested singular directions is capped by the available columns
of `U`.

The `local_enumeration/mutation/` subpackage does not contain another MUTANG
class. It contains scoring and selection only.

### Mutation strategies

`mutation/strategies/` separates methods and shared operations into navigable
modules. Its `__init__.py` exports the complete public API:

- `lpbebo.py`: decoder probability potential and LPBEBO filter;
- `geometry.py`: recommended TANDEM variant A, thesis variant B, and shared
  Jacobian/SVD construction;
- `lams.py`, `tandem.py`, `move.py`, and `random.py`: method-specific filters;
- `composition.py`: identity choices, bounded Cartesian products, sequence
  materialization, and nucleus selection;
- `base.py`: shared interfaces and defaults.

`mutation_potentials.py` and `mutation_filters.py` re-export the same objects so
existing integrations continue to work. New code should import from
`pep_compass.local_enumeration.mutation.strategies`.

```python
from pep_compass.local_enumeration.mutation.strategies import (
    LamsFilter,
    LpbeboFilter,
    MoveFilter,
    TandemFilter,
)
```

A potential receives a parent peptide and a MUTANG map. It does not run SORBES,
enforce a local radius, call a black box, or control an optimization budget.

### Mutation filters

Method modules convert a MUTANG map into selected strings:

- `LpbeboFilter`: decoder probability plus temperature-scaled top-p;
- `LamsFilter`: hard minimum pairwise similarity threshold;
- `TandemFilter`: pairwise TANDEM potential plus top-p;
- `MoveFilter`: norm of the summed single-mutation latent displacements;
- `RandomLeBoFilter`: random walker proposals or randomized MUTANG selection.

`bounded_mutations()` limits the Cartesian product before materialization. It
keeps parent residues available, then removes alternatives from the largest
choice lists until the requested bound is met. This stochastic reduction uses
the run seed.

### Local enumerators

`local_enumeration/local_enumerator.py` owns neighbourhood construction:

| Class | Composition | Common-runner use |
| --- | --- | --- |
| `SamplingMutationLocalEnumerator` | SORBES + canonical MUTANG | baseline LE-BO |
| `SamplingFilteredMutationLocalEnumerator` | SORBES + MUTANG + injected filter | LAMS, TANDEM, MOVE, random controls |
| `FilteredMutationLocalEnumerator` | one center Jacobian + MUTANG + filter | LPBEBO |
| `MutationLocalEnumerator` | one center Jacobian + full canonical MUTANG | available library component |
| `EuclideanWalkerLocalEnumerator*` | Euclidean latent walks | not registered |
| `NormalSamplingLocalEnumerator` | Gaussian latent samples | not registered |

Every enumerator returns `set[str]`. Filtered and sampling enumerators enforce
the Levenshtein radius relative to the center of that local-enumeration call.

## Optimization module

### Optimizer contract

`optimization/optimizer.py` defines `AbstractOptimizer.optimize()` with an
evaluation budget and starting point. Concrete optimizers own algorithm state;
the common runner constructs a fresh optimizer per task so repetitions do not
share scored peptides, trust-region state, or counters.

### LE-BO

`optimization/lebo/local_enumeration_bayesian_optimizer.py` implements the
discrete BO loop. It uses:

- `fingerprints.py` for MAP4 representations;
- `kernel.py` for `TanimotoSimilarityKernel`;
- a BoTorch `SingleTaskGP` and log Expected Improvement;
- a BK-tree for the Levenshtein trust region;
- either Levenshtein or BLOSUM diversity within a selected batch.

The optimizer represents maximization black boxes internally by negating their
scores, so its surrogate and acquisition consistently minimize. GP tensors and
acquisition values use the configured device. Fingerprints and POLI black-box
calls cross a NumPy boundary because those APIs are not Torch-native.

### Baselines

`optimization/baselines/` provides:

- `RandomMutationOptimizer`, optionally filtered by ESM2 PLL;
- `LatentCMAESOptimizer`, used with `HydrAMPBlackBoxWrapper`;
- `SaasboOptimizer`, also operating on a latent black box.

LaMBO2 is provided by the optional `poli-baselines[lambo2]` dependency and is
loaded lazily by the runner. It is not duplicated into the PepCompass package.

### Black boxes and observer

`optimization/black_box/` adapts predictors to the POLI black-box contract.
Registered discrete choices are APEX, BattleAMP, hydrophobicity, and ToxiPep.
`HydrAMPBlackBoxWrapper` decodes latent points before calling a discrete oracle;
it is used by CMA-ES and SAASBO. `NegativeBlackBox` reverses an objective when a
solver requires the opposite direction.

`CSVObserver` records `time`, `sequence`, `score`, and `latent_point`. It can
decode latent inputs when an encoder-decoder is provided. The runner initializes
a fresh observer for every task and stores outputs under the resolved grid
directory.

## Utilities module

- `utils/sequence_utils.py`: one-hot conversion and decoded peptide translation;
- `utils/utils.py`: deterministic seeding and timed sections;
- `utils/blosum_utils.py`: cached BLOSUM loading and pairwise sequence scores.

BLOSUM in LE-BO is an optional within-batch diversity rule. It does not replace
the biological black box and is independent of LAMS/TANDEM/MOVE potentials.

## Scripts and configurations

`scripts/runner/run_optimization.py` is the supported common entry point. It loads
configurations lazily so `--dry-run` does not import optional model stacks.
`scripts/runner/aggregate_optimization_results.py` combines observer trajectories while
retaining source identity. Initialization scripts install external assets.

The older `run_*_optimization_*.py` files remain as comparison references until
result parity is verified. New parameter-only experiments should use child JSON
files in `configs/optimization`, not copied scripts.

## Dependencies

The core dependency groups are declared in `pyproject.toml`. Torch is selected
through one mutually exclusive extra: `cpu`, `cu126`, or `cu128`. Scientific
optimization depends on NumPy, SciPy, BoTorch/GPyTorch, POLI, MAP4-related
packages, and sequence utilities.

Model stacks are not uniform. TensorFlow-based predictors and optional LaMBO2
dependencies may require additional installation. The runner imports the chosen
black box and optimizer only when executing a task, preventing unrelated
optional dependencies from breaking configuration validation.

## Remarks

- Choose one Torch device in the configuration. Geometry follows the
  encoder-decoder device; do not construct collaborating objects on conflicting
  devices.
- For local parallel execution, use one process per task. Assign GPUs with
  `execution.devices`; multiple full model processes on one GPU may exhaust
  memory.
- `maximum_candidates` limits filtered methods. Baseline MUTANG still builds its
  full Cartesian product and can grow rapidly after all-position extraction.
- `SORBESWithoutManifoldAcceleration` and the commented historical multi-walker
  are not production paths.
- `analysis/` may rely on historical outputs or additional environments. It is
  not imported by runtime library code.
- The project does not yet declare a stable semantic-versioned public API.
  Treat underscored helpers, `step_info` internals, and legacy scripts as
  implementation details.

## Validation strategy

The repository declares `pytest` but does not currently contain a committed
test suite or CI workflow. Integration tests used while preparing the stacked
changes were intentionally kept local. A future suite should separate:

1. pure unit tests without model weights: config merge/grid/CSV, BLOSUM, MUTANG,
   candidate bounds, top-p, pair potentials, kernels, and observer shapes;
2. component tests with deterministic fake encoder-decoder and black box:
   SORBES, every enumerator/filter, LE-BO budget/state, output manifests, and
   aggregation;
3. marked model smoke tests: HydrAMP and each oracle, skipped with an explicit
   reason when optional weights or dependencies are missing;
4. runner integration tests: all registry combinations with fakes and selected
   small real-model runs;
5. non-blocking performance benchmarks: Jacobian/SVD, candidate counts,
   fingerprint cache, GP/acquisition time, oracle batching, and GPU memory.

Important regression properties include all-position MUTANG extraction,
candidate-product bounds, minimal top-p prefixes, LAMS global-minimum scoring,
TANDEM pair-order invariance, trust-region limits, seed reproducibility, no
duplicate black-box evaluation, and consistent tensor devices.

## Extension guide

| Change | Correct location |
| --- | --- |
| new latent movement rule | `local_enumeration/sampling_walker.py` |
| new extraction from `U`/`S` | `local_enumeration/mutation_enumerator.py` |
| new mutation score | matching module under `mutation/strategies/` |
| new candidate selection policy | matching module under `mutation/strategies/` |
| new neighbourhood composition | `local_enumeration/local_enumerator.py` |
| new biological objective | predictor under `models/`, adapter under `optimization/black_box/`, runner registry |
| new optimization algorithm | `optimization/` or `optimization/baselines/`, runner registry |
| new parameter sweep | child JSON under `configs/optimization/` |
| new input peptides | CSV selected by `input_csv` |
