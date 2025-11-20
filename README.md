# PepCompass

### Installation

We suggest using [uv](https://docs.astral.sh/uv/) for dependency management. To install the package with dependencies, run:

Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
uv sync
```

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
