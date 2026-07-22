# Optimization runners

## LE-BO and APEX

`scripts/run_lebo_apex.py`

- Runs the APEX benchmark through one configurable LE-BO-family entry point.
- Builds the HydrAMP encoder, tangent-space mutation enumerator, optional SORBES
  walker and method-specific candidate filter, LE-BO optimizer, APEX black box,
  and CSV observer.
- Supports `lebo`, `lpbebo`, `lams`, `tandem`, `move`, `random_walker`, and
  `random_mutang` through `configs/lebo_apex/*.json`.
- Replaces five separate scripts introduced with LAMS, LPBEBO, MOVE, TANDEM,
  and random LE-BO. The configurations hold the actual experimental variants.
- Saves the resolved JSON configuration and the existing observer's CSV
  trajectory under the configured output directory.

Example:

```bash
python scripts/run_lebo_apex.py --config configs/lebo_apex/tandem.json
```

`configs/lebo_apex/base.json`

- Defines shared APEX, encoder, mutation, walker, optimizer, and output settings.
- Is inherited by each method configuration, avoiding script-level duplication.

`configs/lebo_apex/benchmark.json`

- Defines starting peptides and repetition counts shared by all methods.

`configs/lebo_apex/{lebo,lpbebo,lams,tandem,move,random_walker,random_mutang}.json`

- Select the candidate-generation method and override only its parameters.
- They are experiment definitions, not independent runners.

## Existing optimization scripts

`scripts/run_lebo_optimization_{apex,battle,hydro,toxi}.py`

- Run baseline LE-BO against different black boxes.
- Duplicate the experiment setup and differ mainly in oracle and benchmark data.
- The APEX variant overlaps with `run_lebo_apex.py`; the others are not yet
  represented by the unified configuration schema.

`scripts/run_lebo_kappa_optimization_{battle,hydro,toxi}.py`

- Run LE-BO sensitivity experiments for the walker threshold (`kappa`).
- Duplicate the baseline scripts with a parameter sweep and different oracles.

`scripts/run_lebo_cond_0_0_optimization_{battle,hydro}.py`

- Run the HydrAMP condition `[0, 0]` ablation.
- Duplicate the baseline setup with a changed encoder condition.

`scripts/run_random_mutation_optimization.py` and
`scripts/run_random_mutation_optimization_{apex,battleamp,hydro}.py`

- Run random-mutation baselines with oracle-specific setup.
- They are variants of the same baseline but have diverged CLI and defaults.
- `scripts/random_mutation_apex_gpu.sh` is a SLURM wrapper for the APEX variant.

`scripts/run_cmaes_optimization_{battleamp,hydro,toxi}.py`

- Run the CMA-ES baseline against three black boxes.
- Duplicate most orchestration and differ mainly in oracle construction.

`scripts/run_lambo_optimization_{apex,battle,hydro,toxi}.py`

- Run LaMBO against different black boxes.
- These are large, substantially duplicated standalone implementations; some
  also embed their own CSV observer rather than using the shared observer.

`scripts/run_saasbo_optimization.py`

- Runs the SAASBO baseline.
- It is a distinct optimizer, not a wrapper around LE-BO.

`scripts/run_random_esm_apex.py`

- Runs an ESM-based random baseline against APEX.
- It is distinct from both random mutation and the random LE-BO controls.

`scripts/mic_trial_one.py`

- Performs a MIC prediction trial.
- It is an evaluation utility, not an optimization runner.

## RL and cluster wrappers

`scripts/a2c_job_<peptide>.sh` and `scripts/a2c_ext_job_<peptide>.sh`

- Submit per-peptide A2C experiments; the `ext` group uses the extended action
  variant.
- Every file repeats the same SLURM template with a different peptide name.
- They target `scripts/rl_actor_critic_optimizer.py`, which is not present on
  current `dev`, so the wrappers cannot currently run from this branch.

`scripts/a2c_ext_{combo1,combo2,single3,single4}.sh`

- Batch selected extended A2C peptide experiments into individual SLURM jobs.
- They are scheduling wrappers over the same missing A2C runner.

`scripts/rl_job_<peptide>.sh`, `scripts/rl_slurm_array.sh`, and
`scripts/rl_slurm_job.sh`

- Submit per-peptide, array, or single RL experiments.
- They duplicate cluster and optimizer arguments at different scheduling
  granularities.
- They target `scripts/rl_peptide_optimizer.py`, which is not present on current
  `dev`, so these wrappers cannot currently run from this branch.

## Adam's experiment tracking

The history attributed to Adam added separate LE-BO, kappa, condition-ablation,
random-mutation, CMA-ES, and SAASBO scripts and used `CSVObserver` for trajectory
logging. No shared runner, SQLite schema, TensorBoard writer, or Weights & Biases
integration is present in that history or on current `dev`. `sqlalchemy` and
`wandb` are dependencies, but they are not used by these runners.

## Recommended follow-up architecture

Keep one runner with a registry of optimizer, local-enumerator, and black-box
builders. Extend the versioned JSON configuration with oracle and benchmark
selection before replacing the existing non-APEX scripts. Use TensorBoard for
live scalar and distribution views, but keep SQLite as the durable source for
runs, resolved parameters, parent-to-candidate relations, filtering decisions,
and evaluations. CSV should remain an export format. Grid and SLURM launchers
should generate configurations and invoke the same runner rather than duplicate
experiment logic.

The next consolidation should move launchers into `scripts/cluster`, reusable
experiment definitions into `configs`, and one-off analysis utilities into
`scripts/analysis`. Existing scripts should only be removed after parity checks
against the unified runner.
