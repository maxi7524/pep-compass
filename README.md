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

# Example script

```
python scripts/run_lebo_optimization_apex.py
```

# Random mutation with ESM PLL filtering

Install dependencies:

```bash
uv sync
```

The random mutation runner supports ESM2 filtering where mutation candidates are rejected if
`PLL < -0.5` (default threshold). The default model is the lightest ESM2 variant:
`esm2_t6_8M_UR50D`.

Example (KY14, 1400 steps, dry setup command):

```bash
python scripts/run_random_mutation_optimization.py --protein-key KY14 --evaluation-budget 1400 --esm-model-name esm2_t6_8M_UR50D --esm-ppl-threshold -0.5 --device cpu --esm-device cpu
```

To disable ESM filtering:

```bash
python scripts/run_random_mutation_optimization.py --disable-esm-filter
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
scripts/initialization/download_apex_models.sh
```

The script downloads only
`optimization/apex_oracle/APEX_pathogen_models` from
[APEXGo](https://github.com/Yimeng-Zeng/APEXGo) and installs it in
`src/pep_compass/models/apex/APEX_pathogen_models`. Existing weights are never
overwritten.
