# PepCompass

### Installation

We suggest using [uv](https://docs.astral.sh/uv/) for dependency management. To install the package with dependencies, run:

Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create environment

```bash
# CPU only enironment
uv sync
```

or

```bash
# CUDA 12.6
uv sync --extra cu126
```

or

```bash
# CUDA 12.8
uv sync --extra cu128
```

# Optimization runner

```bash
uv run python scripts/run_optimization.py \
  --config configs/optimization/lebo.json
```

The same runner supports parameter grids, multiple starting sequences, Slurm
`srun`, all LE-BO variants, random mutation, CMA-ES, SAASBO, and LaMBO2. See
[`docs/optimization_runners.md`](docs/optimization_runners.md) for the complete
configuration reference. For example, run random mutation with ESM filtering:

```bash
uv run python scripts/run_optimization.py \
  --config configs/optimization/random_mutation_esm.json
```

# Baselines and BlackBoxes

Different baselines and black-boxes needs different packages.

To run toxi:

```
uv sync
uv pip install "tensorflow[and-cuda]==2.20"
uv pip install "numpy==2.3"
```

To run battle:

TBA


### Getting APEX model weights
APEX weights are required for APEX-based evaluation/optimization.

Run the initialization script from the repository root:

```bash
scripts/initialization/download_apex_models_article.sh
```

The script downloads only
`optimization/apex_oracle/APEX_pathogen_models` from
[APEXGo](https://github.com/Yimeng-Zeng/APEXGo) and installs it in
`src/pep_compass/models/apex/APEX_pathogen_models`. Existing weights are never
overwritten.
