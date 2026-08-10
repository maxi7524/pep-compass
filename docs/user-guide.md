# User Guide

This guide documents the public YAML configuration and command-line interface
for composable PepCompass experiments. Internal execution is described in
[Technical Architecture](technical-architecture.md); implementation procedures
are described in [Developer Guide](developer-guide.md).

## Table of Contents

- [User Guide](#user-guide)
  - [Table of Contents](#table-of-contents)
  - [Quick Start](#quick-start)
    - [CLI Command](#cli-command)
    - [Experiment Configuration](#experiment-configuration)
      - [Reproducibility](#reproducibility)
      - [Tracking](#tracking)
      - [Output Files](#output-files)
  - [Parameter Reference](#parameter-reference)
    - [Experiment Parameters](#experiment-parameters)
      - [Input Parameters](#input-parameters)
      - [Output Parameters](#output-parameters)
      - [Grid Parameters](#grid-parameters)
      - [Execution Parameters](#execution-parameters)
    - [Encoder-Decoder Parameters](#encoder-decoder-parameters)
    - [Optimisation Parameters](#optimisation-parameters)
      - [Resource Limits](#resource-limits)
      - [Sequential Steps](#sequential-steps)
      - [Loops](#loops)
      - [Parallel Trajectories](#parallel-trajectories)
      - [Merge Policies](#merge-policies)
    - [Available Components](#available-components)
      - [Encoder-Decoders](#encoder-decoders)
      - [Walkers](#walkers)
      - [Mutation Generators](#mutation-generators)
      - [Filters](#filters)
      - [Selectors](#selectors)
      - [Oracles](#oracles)
  - [Execution Backends](#execution-backends)
    - [Local Execution](#local-execution)
    - [Subprocess Execution](#subprocess-execution)
    - [Slurm Execution](#slurm-execution)
  - [Complete Examples](#complete-examples)
  - [Troubleshooting](#troubleshooting)

## Quick Start

### CLI Command

Run commands from the repository root. The runner resolves relative input and
output paths against the process working directory.

```bash
uv run --extra cu118 python \
  assets/scripts/runner/run_composable_optimization.py \
  --config experiments/configs/reference_lebo.yaml
```

Use `--dry-run` to validate the configuration and print the materialised run
plan without constructing models or executing optimisation:

```bash
uv run --extra cu118 python \
  assets/scripts/runner/run_composable_optimization.py \
  --config experiments/configs/reference_lebo.yaml \
  --dry-run
```

The complete CLI is:

```text
--config PATH                    required YAML or JSON configuration
--dry-run                        validate and print the run plan
--resume                         skip runs with completed result files
--continue-on-error              continue after a failed run
--run-index INDEX                execute one plan index; may be repeated
--backend local|subprocess|slurm override the configured backend
--max-workers INTEGER            override subprocess worker count
--slurm-script PATH              output script required by the Slurm backend
--working-directory PATH         base for relative input and output paths
```
> Remark:
> Validation configurations under `experiments/configs/validation/` isolate walkers, mutation generators, filters, oracles, loops, parallel execution, ESM and trust-region behaviour. Use these files before running the full reference configuration.

### Experiment Configuration

The following configuration contains every top-level section and the supported
experiment controls. Strategy-specific parameters are documented under
[Available Components](#available-components).

```yaml
experiment:
  name: example
  seed: 1234
  input:
    csv:
      path: data/peptides/peptides_smoke.csv
      sequence_column: sequence
      repetitions_column: repetitions
  output:
    directory: experiments/results/example
  tracking:
    level: normal
    max_depth: null
    store_latents: false
    store_fields: false
  execution:
    backend: subprocess
    max_workers: 1
    slurm:
      job_name: pep-example
      partition: gpu
      time: "24:00:00"
      gres: gpu:1
      cpus_per_task: 4
      mem: 16G
  grid:
    optimization.steps.0.loop.iterations: [1, 2]

encoder_decoder:
  method: hydramp
  device: cpu # or cuda
  parameters:
    jacobian_mode: approx
    jacobian_eps: 0.001
    field_eps: 0.001

optimization:
  limits:
    oracle_calls: 10
    generated_candidates: 100000
  steps:
    - loop:
        iterations: 1
        steps:
          - parallel:
              replicas: 10
              execution: auto
              merge: concatenate
              steps:
                - loop:
                    iterations: 10
                    steps:
                      - walker:
                          method: sorbes
                          parameters:
                            horizontal_threshold: 0.001
                            time_step: 0.01
                            max_horizontal_update_norm: 0.1
                            vertical_movement: false
                      - mutation_generator:
                          method: mutang
                          parameters:
                            max_len: 25
                            direction_significance_threshold: 0.000001
                            min_number_of_directions: 5
                            token_threshold: 0.1
          - filter:
              method: deduplicate
              parameters: {key: sequence_and_latent}
          - oracle:
              method: hydrophobicity
              parameters: {}
```

Every list entry under `steps` must contain exactly one operation key:
`loop`, `parallel`, `walker`, `mutation_generator`, `filter` or `oracle`.

#### Reproducibility

`experiment.seed` is the base seed. A stable plan index is added for each grid
variant, input sequence and repetition. Parallel branches receive seeds derived
from the run seed and their branch index. The run manifest records the resolved
run seed.

`repetitions` creates independent runs. It does not duplicate candidates inside
one optimisation batch.

#### Tracking

`tracking.level` accepts `short`, `normal` or `all`:

- `short` records oracle executions and evaluated candidates;
- `normal` records step summaries, sizes, timings and counters;
- `all` additionally records candidates produced by every enabled step.

`max_depth` limits collection by execution-tree depth without changing the
executed steps. `store_latents` serialises candidate latent origins.
`store_fields` serialises algorithm fields and may produce large files when
SORBES matrices are present.

Tracking rows include nested loop indices, parallel branch names and parallel
branch indices. Replicas are named `replica_000`, `replica_001`, and so on.

#### Output Files

Each run directory contains:

```text
runs/run_00000/
  result.json
  candidates.csv
  latent_origins.pt
  tracking/
    steps.csv
    candidates.csv
```

Multiple grid variants add `variants/variant_XXXXX/` above `runs/`. The output
root also contains `run_manifest.csv`. `result.json` stores status, input
identity, final candidate count and an optional objective summary. Experiments
without an oracle produce `null` objective, best sequence and best score.

## Parameter Reference

### Experiment Parameters

`experiment.name` identifies the experiment. `seed`, `input`, `output`,
`tracking`, `execution` and `grid` control run materialisation and persistence.

#### Input Parameters

Configure exactly one input form.

Inline input:

```yaml
input:
  sequences: [FLYKWWIRIGRLKL, ACDEFGHIK]
  repetitions: 2
```

CSV input:

```yaml
input:
  csv:
    path: data/peptides/peptides.csv
    sequence_column: sequence
    repetitions_column: repetitions
```

`path` may be absolute or relative to the runner working directory. The
sequence column is required. If the repetitions column is absent from a row,
that row defaults to one run.

#### Output Parameters

`output.directory` is optional. Omitting it disables persisted experiment
results. Relative output paths use the same working directory as relative
inputs.

#### Grid Parameters

`experiment.grid` maps existing dotted paths rooted at `optimization` to lists
of values. The runner constructs their Cartesian product. For example, two
loop counts and three filter methods create six variants.

```yaml
grid:
  optimization.steps.0.loop.iterations: [1, 2]
  optimization.steps.0.loop.steps.2.filter.method: [lpbebo, lams, tandem]
```

The value in the base tree is replaced in every materialised variant. Keep it
equal to the first grid value so the configuration remains readable when the
grid is removed.

#### Execution Parameters

`execution.backend` accepts `local`, `subprocess` or `slurm`.
`execution.max_workers` must be a positive integer. The nested `slurm` mapping
accepts `job_name`, `partition`, `time`, `gres`, `cpus_per_task` and `mem`.

### Encoder-Decoder Parameters

The current built-in method is `hydramp`.

```yaml
encoder_decoder:
  method: hydramp
  device: cuda
  parameters:
    jacobian_mode: approx
    default_condition: [1.0, 1.0]
    temp: 1.0
    jacobian_eps: 0.001
    field_eps: 0.001
```

`jacobian_eps` and `field_eps` are required. `jacobian_mode` accepts `strict`
or `approx`. `default_condition` must contain two values. `temp` controls the
decoder softmax temperature.

### Optimisation Parameters

#### Resource Limits

`oracle_calls` limits evaluated sequences. The oracle truncates its input to
the remaining budget. `generated_candidates` counts candidates reported by
mutation generators and requests termination after the configured threshold is
reached. Use `null` to disable either limit.

#### Sequential Steps

Steps in one list execute in order. Each receives the `CandidateBatch` returned
by the previous step. The engine does not insert missing walkers, generators,
filters or oracles.

#### Loops

`loop.iterations` is a non-negative integer. `loop.steps` may contain any
supported operations, including nested loops and parallel sections. A loop
stops early when the global stop state is requested or its candidate batch is
empty.

#### Parallel Trajectories

Use replicas when every trajectory has the same definition:

```yaml
- parallel:
    replicas: 10
    execution: auto
    merge: concatenate
    steps:
      - loop:
          iterations: 10
          steps: [...]
```

The runtime creates `replica_000` through `replica_009`, derives an RNG for
each index and merges outputs in replica order.

Use explicit branches when trajectories differ:

```yaml
- parallel:
    execution: auto
    merge: concatenate
    branches:
      - name: conservative
        steps: [...]
      - name: exploratory
        steps: [...]
```

`replicas + steps` and `branches` are mutually exclusive. `execution` accepts
`auto`, `concurrent` or `sequential`. `auto` selects the available parallel
implementation without making the configuration depend on that implementation.
The current runtime uses concurrent execution as its automatic fallback.

#### Merge Policies

`concatenate` is the implemented merge policy. It preserves branch order and
duplicates. Fields missing from selected branches are represented with
validity masks. `interleave`, `select_best` and `weighted_sample` are reserved
extension points and currently raise `NotImplementedError`.

### Available Components

#### Encoder-Decoders

- `hydramp`: peptide encoder-decoder with strict or approximate Jacobians.

#### Walkers

- `sorbes`: accepts `horizontal_threshold`, `time_step`,
  `max_horizontal_update_norm` and optional `vertical_movement`.

#### Mutation Generators

- `mutang`: accepts `max_len`, `direction_significance_threshold`,
  `min_number_of_directions`, `token_threshold` and optional `alphabet`.

MUTANG requires the singular-value and left-vector fields produced by SORBES.
For HydrAMP, `max_len` must match the model sequence length of 25; it is not a
candidate limit.

#### Filters

- `lpbebo`: decoder log-probability and top-p selection;
- `lams`: tangent-space similarity threshold;
- `tandem`: pairwise tangent similarity and top-p selection;
- `move`: approximate latent-displacement selection;
- `random_walker`: random mutation proposal control;
- `random_mutang`: random selection over the MUTANG product;
- `esm_plausibility`: ESM pseudo-log-likelihood threshold or top-k selection.

Mutation-choice filters consume `mutation.parent_sequence` and
`mutation.options`, so they must follow MUTANG. Their exact parameters are shown
in `experiments/configs/validation/mutation_filters_smoke.yaml` and
`esm_smoke.yaml`.

#### Selectors

- `deduplicate`: `key` is `sequence` or `sequence_and_latent`;
- `robot`: requires `objective`; supports `batch_size`, `maximize`,
  `diversity_threshold`, `acquisition_batch_size`, `standardize` and `device`;
- `trust_region`: requires `objective`, `initial_radius` and `geometry` set to
  `sequence` or `latent`;
- `trust_region_update`: updates the centre and radius after the matching
  oracle score is attached.

#### Oracles

Built-in methods are `hydrophobicity`, `apex`, `battleamp`, `toxipep`, `eipred`
and `mbc_attention`. Every oracle accepts optional POLI controls and
`evaluation_batch_size`. Model-specific parameters are validated before model
construction. BattleAMP and MBC-Attention require TensorFlow in the runtime
environment.

## Execution Backends

### Local Execution

`local` executes selected runs in the current Python process and reuses the
constructed encoder-decoder. Use it for debugging one run.

### Subprocess Execution

`subprocess` starts isolated interpreters. `max_workers` controls the number of
simultaneous runs. Every worker receives the absolute config path and working
directory.

### Slurm Execution

`slurm` writes an array script but does not submit it. Pass the destination:

```bash
uv run --extra cu118 python \
  assets/scripts/runner/run_composable_optimization.py \
  --config experiments/configs/reference_lebo.yaml \
  --backend slurm \
  --slurm-script experiments/jobs/reference_lebo.sh
```

The script contains `#SBATCH --chdir=<working-directory>` and uses plan indices
as array indices. Submit the generated file with the site-specific Slurm
command.

## Complete Examples

- `experiments/configs/composable_example.yaml`: compact composable tree.
- `experiments/configs/reference_lebo.yaml`: replicated SORBES–MUTANG
  trajectories followed by LPBEBO, ROBOT, APEX and trust-region updates.
- `experiments/configs/validation/walker_smoke.yaml`: SORBES parameter grid.
- `experiments/configs/validation/mutation_generator_smoke.yaml`: MUTANG grid.
- `experiments/configs/validation/mutation_filters_smoke.yaml`: mutation filter
  grid.
- `experiments/configs/validation/oracle_smoke.yaml`: oracle grid.
- `experiments/configs/validation/parallel_smoke.yaml`: execution-mode grid.
- `experiments/configs/validation/latent_trust_region_smoke.yaml`: complete
  latent trust-region flow.

## Troubleshooting

`must contain exactly one operation key` means one item under `steps` contains
multiple sibling keys, usually because `iterations` or `steps` is indented at
the wrong level.

`Grid path does not exist` means the dotted grid path does not identify a leaf
already present in the base optimisation tree.

`MUTANG requires walker.singular_values and walker.left_vectors` means MUTANG
was executed without an earlier SORBES step in the same data flow.

A completed result with zero candidates and `null` objective fields is valid
when filters remove the entire pool before oracle evaluation.
