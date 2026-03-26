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

1. Clone the APEXGo repository:
```bash
git clone https://github.com/Yimeng-Zeng/APEXGo.git
```
2. Within that repository, locate the directory:
   `optimization/apex_oracle/APEX_pathogen_models`
3. Copy that directory into this project at:
   `src/pep_compass/models/apex/APEX_pathogen_models`

After copying, you should have:
`src/pep_compass/models/apex/APEX_pathogen_models/<model_files>`
