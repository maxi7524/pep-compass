# Optimization runners

## Environment setup

Run commands from the repository root. Create the project environment and
install the APEX weights once:

```bash
uv sync --extra cpu
scripts/initialization/download_apex_models.sh
```

Use the project environment either through `uv run` or by activating it:

```bash
source .venv/bin/activate
```

The default configuration uses `cuda:0`. For a CPU run, pass `--device cpu`.
CPU support depends on the underlying HydrAMP and APEX model stack; a dry run
does not load either model and can be used to validate input files and grids.

## LE-BO APEX runner

The supported entry point is `scripts/run_lebo_apex.py`. It constructs, in
order:

1. the HydrAMP encoder-decoder;
2. MUTANG and, for applicable methods, the SORBES walker;
3. the selected candidate filter;
4. the LE-BO optimizer;
5. the APEX black box and CSV trajectory observer.

Run one predefined experiment:

```bash
uv run python scripts/run_lebo_apex.py \
  --config configs/lebo_apex/tandem.json
```

Available method configurations are `lebo.json`, `lpbebo.json`, `lams.json`,
`tandem.json`, `move.json`, `random_walker.json`, and `random_mutang.json`.

Useful command-line overrides:

```bash
uv run python scripts/run_lebo_apex.py \
  --config configs/lebo_apex/lams.json \
  --device cuda:0 \
  --output results/lams_trial \
  --budget 200 \
  --seed 1234
```

Use `--dry-run` before a large experiment. It validates the CSV, expands the
grid, and writes resolved configurations without loading the models:

```bash
uv run python scripts/run_lebo_apex.py \
  --config configs/lebo_apex/template.json \
  --output /tmp/lebo_grid_check \
  --dry-run
```

## Input sequences

`input_csv` in `base.json` points to `peptides.csv`. The CSV has three columns:

```csv
name,sequence,repetitions
middle-1,FLYKWWIRIGRLKL,4
new-peptide,ACDEFGHIK,3
```

- `name` is a unique label used in output filenames.
- `sequence` is the starting peptide passed to `optimizer.optimize` as
  `starting_point`. Optimization generates and evaluates candidates around this
  sequence.
- `repetitions` is optional and defaults to `1`. It controls how many
  independent runs are made for that starting sequence. Repetitions receive
  different seeds; the same seed schedule is reused across grid variants so
  their results can be compared.

To add sequences, append rows to a CSV. To use another dataset, copy the file
and override the path in a configuration located next to `base.json`:

```json
{
  "extends": "base.json",
  "input_csv": "my_peptides.csv",
  "method": "lebo",
  "output_path": "./results/my_lebo_run"
}
```

Paths inherited from `base.json` are resolved relative to that configuration
directory.

## Configuration and parameter grids

`base.json` contains one complete set of defaults. A method configuration uses
`extends` and overrides only values that differ:

```json
{
  "extends": "base.json",
  "method": "tandem",
  "output_path": "./results/tandem",
  "filter": {
    "top_p": 0.9,
    "temperature": 1.0
  }
}
```

Copy `configs/lebo_apex/template.json` to define a grid. The `grid` object maps
dotted configuration paths to non-empty lists:

```json
{
  "extends": "base.json",
  "output_path": "./results/tandem_grid",
  "grid": {
    "method": ["tandem"],
    "filter.top_p": [0.6, 0.8, 0.9],
    "filter.temperature": [0.5, 1.0],
    "mutation.token_threshold": [0.05, 0.1],
    "walker.horizontal_threshold": [0.05, 0.1]
  }
}
```

Every value in `grid` must be a list, including a parameter with only one
choice. The runner creates the Cartesian product, so the example produces
`3 × 2 × 2 × 2 = 24` variants. Each variant runs every CSV row according to its
`repetitions` value. Therefore the total number of optimizer runs is:

```text
number of grid variants × sum(repetitions in the input CSV)
```

Parameters that are themselves vectors use a list of complete values:

```json
"encoder.default_condition": [[1.0, 1.0], [0.0, 0.0]],
"apex.mic_bacteria": [[1, 2, 3], [1]]
```

Do not convert normal vector parameters in `base.json` into grids. Only values
inside the explicit `grid` object are expanded.

## Output and aggregation

The output root contains:

```text
results/tandem_grid/
├── grid_manifest.csv
├── grid_0000/
│   ├── resolved_config.json
│   └── <APEX black-box name>/*.csv
└── grid_0001/
    ├── resolved_config.json
    └── <APEX black-box name>/*.csv
```

- `grid_manifest.csv` maps every `grid_id` to its parameter combination and
  output directory.
- `resolved_config.json` records the complete configuration used by a variant.
- Each observer CSV is one trajectory and contains timestamp, peptide sequence,
  APEX score, and latent point.

Combine every trajectory into one CSV with:

```bash
uv run python scripts/aggregate_lebo_results.py results/tandem_grid \
  --output results/tandem_grid.csv
```

The combined file adds `grid_id`, `method`, and `source_file`, allowing it to be
joined with `grid_manifest.csv` and grouped in pandas, R, or another analysis
tool. The original per-run files remain unchanged.

## Existing standalone runners

The following scripts predate the unified runner and still duplicate setup:

- `run_lebo_optimization_{apex,battle,hydro,toxi}.py`: baseline LE-BO by oracle;
- `run_lebo_kappa_optimization_{battle,hydro,toxi}.py`: walker-threshold sweep;
- `run_lebo_cond_0_0_optimization_{battle,hydro}.py`: encoder-condition ablation;
- `run_random_mutation_optimization*.py`: random-mutation baselines;
- `run_cmaes_optimization_*.py`: CMA-ES baselines;
- `run_lambo_optimization_*.py`: standalone LaMBO implementations;
- `run_saasbo_optimization.py`: SAASBO baseline;
- `run_random_esm_apex.py`: ESM random baseline.

They are not configuration files for `run_lebo_apex.py`. They should be moved
to the shared runner only after their oracle-specific behavior is represented
and compared against the original scripts.

The `a2c_*.sh` wrappers reference `scripts/rl_actor_critic_optimizer.py`, and the
`rl_*.sh` wrappers reference `scripts/rl_peptide_optimizer.py`. Neither Python
runner is present on current `dev`, so those wrappers cannot currently run from
this branch.
