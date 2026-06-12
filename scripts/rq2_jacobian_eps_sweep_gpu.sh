#!/bin/bash
#SBATCH --job-name=rq2-eps
#SBATCH --partition=common
#SBATCH --qos=kjurasz
#SBATCH --gres=gpu:rtx5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=logs/rq2_eps_%j.out
#SBATCH --error=logs/rq2_eps_%j.err

set -euo pipefail
cd /home/kjurasz/pep-compass
mkdir -p logs

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found; refusing CPU fallback." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

PYTHON=/home/kjurasz/pep-compass/.venv/bin/python
export RQ2_EPS_N=${RQ2_EPS_N:-500}
export RQ2_EPS_SEED=${RQ2_EPS_SEED:-0}
export RQ2_DEVICE=${RQ2_DEVICE:-cuda}
export THESIS_FIG_DIR=/home/kjurasz/pep-compass/results/thesis_figs
export PYTHONUNBUFFERED=1
mkdir -p "${THESIS_FIG_DIR}"

echo "RQ2_EPS_N=${RQ2_EPS_N}  RQ2_EPS_SEED=${RQ2_EPS_SEED}  RQ2_DEVICE=${RQ2_DEVICE}"
time "${PYTHON}" analysis/scripts/thesis_figures/fig_rq2_jacobian_eps_sweep.py
echo "=== outputs ==="
ls -la results/data/all_in/_cache/_thesis_rq2_eps*.csv
ls -la "${THESIS_FIG_DIR}/rq2_eps_sensitivity.pdf"
