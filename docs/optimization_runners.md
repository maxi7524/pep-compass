# Optimization runner

`scripts/runner/run_optimization.py` is the common entry point for optimization
experiments. A JSON configuration selects the optimizer, black box, LE-BO
variant, input peptides, parameter grid, and execution backend. The runner
materializes every run before loading a model, which makes a dry run sufficient
to validate the complete grid and Slurm commands.

The entry point only parses CLI overrides and starts prepared tasks. Supporting
modules in the same directory separate responsibilities:

- `configuration.py` loads inherited JSON, validates it, expands grids, reads
  peptide CSV files, and materializes task manifests;
- `builders.py` constructs black boxes, encoders, local enumerators, and
  optimizers;
- `execution.py` executes one task and provides the local-process and Slurm
  backends;
- `aggregate_optimization_results.py` combines legacy observer CSV files and
  accepts both the historical `method` field and `candidate_strategy`.

## Quick start

Run commands from the repository root:

```bash
uv sync --extra cu126
bash scripts/initialization/download_apex_models.sh
uv run python scripts/runner/run_optimization.py \
  --config configs/optimization/experiments/lebo/lebo.json \
  --dry-run
uv run python scripts/runner/run_optimization.py \
  --config configs/optimization/experiments/lebo/lebo.json
```

> Use `--extra cpu`, `--extra cu126`, or `--extra cu128` to select the Torch
build. The configured `device` must match that environment.

> A dry run does not
load HydrAMP, APEX, or another predictor.

## Execution workflow

The runner performs these operations:

1. Recursively loads `extends` files and merges child values over parent values.
2. Resolves `input_csv` relative to the JSON file that defines it.
3. Expands the explicit `grid` object as a Cartesian product.
4. Combines every grid variant with every CSV row and its `repetitions` value.
5. Writes resolved configurations, task JSON files, and manifests.
6. On the `local` backend, executes one task in-process or multiple tasks in
   isolated Python processes.
7. On the `srun` backend, starts one isolated `srun` step per task and limits
   concurrent steps to `execution.max_parallel_runs`.
8. For each task, builds the selected black box and optimizer and calls it with
   the task sequence and seed. LE-BO writes the structured `tracking/` report;
   other optimizers retain the legacy `CSVObserver` output.

The number of optimizer runs is:

```text
number of grid variants × sum(repetitions in input CSV)
```

`seed` is the base seed. Task `N` receives `seed + N`; therefore every grid
variant uses a deterministic seed schedule recorded in `run_manifest.csv`.

## Input CSV

`input_csv` points to a CSV containing starting peptides:

```csv
name,sequence,repetitions
middle-1,FLYKWWIRIGRLKL,4
new-peptide,ACDEFGHIK,2
```

- `name` is a unique experiment label.
- `sequence` is passed as `starting_point` to the optimizer.
- `repetitions` is optional and defaults to `1`; it creates independent runs
  with distinct seeds for the same sequence.

Add sequences by adding CSV rows, or copy `peptides.csv` and point a child
configuration at the new file:

```json
{
  "extends": "../../base.json",
  "input_csv": "../../peptides/my_peptides.csv",
  "output_path": "./results/my_experiment"
}
```

## Constructing a configuration

`configs/optimization/base.json` is the complete executable configuration.
Copy a small child configuration for a single experiment, or copy
`configs/optimization/template.json` when constructing a grid. Values outside
`grid` are normal values. Values inside `grid` must always be non-empty lists.

```json
{
  "extends": "../../base.json",
  "output_path": "./results/tandem_grid",
  "optimizer": {
    "lebo": {"candidate_strategy": "tandem"}
  },
  "grid": {
    "filter.top_p": [0.6, 0.8, 0.9],
    "filter.temperature": [0.5, 1.0],
    "mutation.token_threshold": [0.05, 0.1]
  }
}


# This produces `3 × 2 × 2 = 12` variants. 
```


Vector-valued parameters use a list of
complete vectors:
```json
{
  "grid": {
    "encoder.default_condition": [[1.0, 1.0], [0.0, 0.0]],
    "black_box.apex.mic_bacteria": [[1, 2, 3], [1]]
  }
}
```

CLI overrides the corresponding top-level value from a grid so a single
value is used:

```bash
uv run python scripts/runner/run_optimization.py \
  --config configs/optimization/template.json \
  --device cuda:0 \
  --budget 200 \
  --seed 1234 \
  --output results/check
```

## Parameter groups

### General

- `input_csv`: starting-peptide table.
- `output_path`: root for manifests, task files, and optimization results.
- `device`: Torch device passed to compatible models and optimizers.
- `evaluation_budget`: budget passed to `optimizer.optimize`.
- `seed`: base task seed; `null` selects the current Unix time once.
- `optimizer.lebo.candidate_strategy`: LE-BO candidate strategy. Available
  values are `lebo`, `lpbebo`, `lams`, `tandem`, `move`, `random_walker`, and
  `random_mutang`.

### Tracking

The `tracking.level` setting controls how much LE-BO provenance is persisted:

- `short` writes iteration summaries and peptides evaluated by the black box;
- `normal` additionally writes the generating trajectories of evaluated
  peptides;
- `full` additionally writes every locally accepted candidate that could enter
  optimizer selection.

Rejected peptide strings and duplicate generation events are not stored. Their
effect is visible through the aggregate `generated_count`, `accepted_count`,
and `rejected_count` columns. `tracking.store_latents` controls latent encoding
for candidate rows; disable it when only sequence provenance is needed.

Each LE-BO task writes a directory below
`<output_path>/<grid_id>/tracking/<experiment_id>/` containing:

- `tracking_metadata.json`: objective name, mathematical meaning, direction,
  black-box parameters, candidate strategy, and tracking settings;
- `iteration_statistics.csv`: one row per optimizer iteration;
- `evaluations.csv`: raw black-box objective values with their meaning and
  optimization direction;
- `candidates.csv`: trajectory provenance for `normal` and `full` tracking.

`trajectory_path` is a JSON list stored inside one CSV field. It contains the
decoded sequence path from the iteration centre to the candidate. A candidate
sequence is written once; repeated generation of the same sequence is ignored.
`run_id`, `iteration_id`, `trajectory_id`, `step_id`, and `candidate_id` remain
available after concatenating results from multiple tasks.

### Execution

- `execution.backend`: `local` or `srun`.
- `execution.max_parallel_runs`: maximum concurrent local processes or `srun`
  steps. A value of `1` keeps local execution in the runner process.
- `execution.devices`: optional device names assigned round-robin to task
  configurations, for example `["cuda:0", "cuda:1"]`.
- `execution.srun.command`: normally `srun`.
- `execution.srun.arguments`: arguments placed between `srun` and the Python
  command, for example `--exclusive`, `--partition=gpu`, or `--gres=gpu:1`.

The equivalent CLI invocation is:

```bash
uv run python scripts/runner/run_optimization.py \
  --config configs/optimization/template.json \
  --execution srun \
  --max-parallel-runs 4 \
  --devices cuda:0 \
  --srun-argument=--exclusive \
  --srun-argument=--gres=gpu:1
```

Each step executes the same runner with a generated `--task-file`; there is no
second Slurm-specific experiment implementation. Scheduler, account, partition,
CPU, memory, and GPU arguments remain cluster-specific and belong in
`execution.srun.arguments` or CLI overrides. `srun_template.json` is a copyable
example, not a cluster policy.

For local multi-GPU execution, set `execution.backend` to `local`, set
`max_parallel_runs` to the desired process count, and list the available
devices. Do not start several model processes on one GPU unless its memory is
sufficient.

### Black box

- `black_box.name`: `apex`, `battleamp`, `hydrophobicity`, or `toxipep`.
- `black_box.common.batch_size`: optional POLI evaluation batch size.
- `black_box.common.parallelize`: POLI black-box process parallelism.
- `black_box.common.num_workers`: worker count used when parallelism is enabled.
- `black_box.common.force_isolation`: force POLI evaluation isolation.
- `black_box.apex.mic_aggregate`: APEX MIC aggregation (`mean` or `max` in the
  current black-box implementation).
- `black_box.apex.mic_bacteria`: pathogen-model indices.
- `black_box.hydrophobicity.scale`: hydrophobicity scale name.

Only the subsection matching `black_box.name` is used. APEX requires model
weights installed by `scripts/initialization/download_apex_models.sh`.

### Optimizer

- `optimizer.name`: `lebo`, `random_mutation`, `cmaes`, `saasbo`, or `lambo2`.
- `optimizer.lebo.*`: LE-BO/TuRBO initialization, acquisition, diversity, and
  trust-region parameters.
- `optimizer.random_mutation.esm_*`: optional ESM model, threshold, device, and
  resampling limit. A `null` model disables ESM filtering.
- `optimizer.cmaes.*`: population size, initial sigma, and constraint penalty.
- `optimizer.saasbo.*`: batch size, NUTS warmup, samples, thinning, and latent
  dimension.
- `optimizer.lambo2.iterations`: LaMBO2 solver iterations.
- `optimizer.lambo2.negate_objective`: explicitly wrap the selected black box
  in `NegativeBlackBox`; `lambo2.json` enables this to reproduce the historical
  APEX script. LaMBO2 additionally requires the optional
  `poli-baselines[lambo2]` dependencies; the runner emits a direct error if they
  are absent.

CMA-ES and SAASBO wrap the selected discrete black box in
`HydrAMPBlackBoxWrapper`. LE-BO builds its own encoder for local enumeration.
Random mutation and LaMBO2 operate on discrete peptide strings.

### Encoder

- `encoder.jacobian_mode`: HydrAMP exact or approximate Jacobian mode.
- `encoder.default_condition`: two-element HydrAMP conditioning vector.
- `encoder.temperature`: decoder sampling temperature.
- `encoder.jacobian_eps`: finite-difference Jacobian epsilon.
- `encoder.field_eps`: vector-field epsilon.

### MUTANG

- `mutation.max_len`: padded sequence length used by tangent-space enumeration.
- `mutation.direction_significance_threshold`: singular-value significance
  threshold.
- `mutation.min_number_of_directions`: minimum number of SVD directions.
- `mutation.token_threshold`: minimum absolute direction component for a residue
  proposal.

### SORBES walker

- `walker.horizontal_threshold`: tangent-space singular-value threshold.
- `walker.time_step`: Brownian step time.
- `walker.max_horizontal_update_norm`: horizontal step norm cap.
- `walker.vertical_movement`: enable the vertical component.

### Local enumeration and candidate strategies

- `local_enumeration.walker_trajectories`: SORBES trajectories per center.
- `local_enumeration.walk_time_budget`: simulated time per trajectory.
- `local_enumeration.max_neighbour_levenshtein`: final radius around the center.
- `filter.maximum_candidates`: cap applied before Cartesian products expand.
- `filter.top_p` and `filter.temperature`: LPBEBO, TANDEM, and MOVE nucleus
  selection.
- `filter.similarity_threshold`: LAMS hard acceptance threshold.
- `filter.selection_fraction`: random-MUTANG retained probability mass.
- `filter.maximum_positions` and `filter.residues_per_position`: random-walker
  proposal size.

`optimizer.lebo.candidate_strategy` selects `lebo`, `lpbebo`, `lams`, `tandem`,
`move`, `random_walker`, or `random_mutang`. The runner builds canonical SORBES
and MUTANG, then inserts the selected filter between MUTANG and the local
enumerator. LPBEBO is the only single-Jacobian path; the other strategies use
the SORBES trajectory loop.

## Configurations replacing standalone scripts

- `lebo.json` plus a `black_box.name` override replaces
  `run_lebo_optimization_{apex,battle,hydro,toxi}.py`.
- `lebo_kappa.json` replaces the three `run_lebo_kappa_optimization_*.py`
  scripts with one threshold grid.
- `lebo_condition_0_0.json` replaces both
  `run_lebo_cond_0_0_optimization_*.py` scripts.
- `random_mutation.json` and `random_mutation_esm.json` replace the discrete
  random-mutation scripts, with the oracle selected independently.
- `cmaes.json`, `saasbo.json`, and `lambo2.json` replace their oracle-specific
  setup scripts; change or grid `black_box.name` instead of copying Python.
- `lams.json`, `lpbebo.json`, `tandem.json`, `move.json`, `random_walker.json`,
  and `random_mutang.json` expose the methods migrated from `rl_trials`.

The old scripts remain present for comparison until result parity is verified.
The A2C and RL shell wrappers are not represented as optimizer configs because
their referenced Python entry points are absent on `dev`.

## Output and aggregation

```text
results/example/
├── grid_manifest.csv
├── run_manifest.csv
├── tasks/
│   └── task_000000.json
└── grid_0000/
    ├── resolved_config.json
    └── <black-box-name>/*.csv
```

- `grid_manifest.csv` maps grid IDs to parameter combinations.
- `run_manifest.csv` records every sequence, repetition, seed, output, and task.
- `tasks/*.json` is the exact serialized unit executed locally or by `srun`.
- `resolved_config.json` is the complete configuration after inheritance and
  grid substitution.
- non-LE-BO optimizers retain observer CSV files with time, sequence, score,
  and an optional latent point. LE-BO does not duplicate evaluations in a
  black-box-named directory; its complete result is under `tracking/`.

Aggregate trajectories without modifying the original files:

```bash
uv run python scripts/runner/aggregate_optimization_results.py results/example \
  --output results/example.csv
```

The aggregate adds `grid_id`, `optimizer`, `black_box`, `candidate_strategy`, and
`source_file`. Join it with `grid_manifest.csv` for parameter-level analysis or
with `run_manifest.csv` for starting sequence, repetition, and seed metadata.
