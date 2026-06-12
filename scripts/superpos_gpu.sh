#!/bin/bash
#SBATCH --job-name=superpos
#SBATCH --partition=common
#SBATCH --qos=kjurasz
#SBATCH --gres=gpu:rtx5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=logs/superpos_%j.out
#SBATCH --error=logs/superpos_%j.err

# First-order net-displacement (superposition) filter at scale (forward passes only: encode only
# the single mutants per parent, score the full candidate set by vector sums). Validates the
# rho~0.99 accuracy and outlier-detection AUROC on the 845-peptide MUTANG+ set.
#   sbatch --export=ALL,SP_INPUT=_thesis_mutangplus_pogs.parquet,SP_OUTPUT=_thesis_superpos_filter_mutangplus.parquet scripts/superpos_gpu.sh

set -euo pipefail
cd /home/kjurasz/pep-compass
mkdir -p logs
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found; refusing CPU fallback." >&2; exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
PYTHON=/home/kjurasz/pep-compass/.venv/bin/python
export SP_INPUT=${SP_INPUT:-_thesis_rq5_distances_pogs.parquet}
export SP_OUTPUT=${SP_OUTPUT:-_thesis_superpos_filter.parquet}
export PYTHONUNBUFFERED=1
echo "SP_INPUT=${SP_INPUT} SP_OUTPUT=${SP_OUTPUT}"
time "${PYTHON}" analysis/scripts/thesis_figures/_exp_superpos_filter.py
echo "=== done ==="; ls -la results/data/all_in/_cache/${SP_OUTPUT}
