#!/bin/bash
#SBATCH --job-name=rq2-jac
#SBATCH --partition=common
#SBATCH --qos=kjurasz
#SBATCH --gres=gpu:rtx5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rq2_jac_%j.out
#SBATCH --error=logs/rq2_jac_%j.err

# Large-scale validation of section 5.2.3:
# compare the production finite-difference Jacobian (epsilon = 5e-2) against
# the autograd-exact Jacobian on >= 5000 peptides sampled from peptides_apex.csv,
# at production thresholds kappa = 1e-3, theta_mut = 1e-6.
#
# Submit:
#   sbatch scripts/rq2_jacobian_largescale_gpu.sh
#   # Larger sample:
#   sbatch --export=ALL,RQ2_N=10000 scripts/rq2_jacobian_largescale_gpu.sh
#   # Override the finite-difference step:
#   sbatch --export=ALL,RQ2_EPS=1e-3 scripts/rq2_jacobian_largescale_gpu.sh
#
# Outputs land in results/data/all_in/_cache/ so the downstream thesis-figure
# scripts pick them up automatically:
#   _thesis_rq2_eps_prod_large.csv
#   _thesis_rq2_eps_prod_large_summary.json

set -euo pipefail
cd /home/kjurasz/pep-compass
mkdir -p logs

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found; refusing CPU fallback." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

PYTHON=/home/kjurasz/pep-compass/.venv/bin/python

export RQ2_N=${RQ2_N:-5000}
export RQ2_SEED=${RQ2_SEED:-0}
export RQ2_EPS=${RQ2_EPS:-5e-2}
export RQ2_DEVICE=${RQ2_DEVICE:-cuda}
export PYTHONUNBUFFERED=1

echo "RQ2_N=${RQ2_N}  RQ2_SEED=${RQ2_SEED}  RQ2_EPS=${RQ2_EPS}  RQ2_DEVICE=${RQ2_DEVICE}"
time "${PYTHON}" analysis/scripts/thesis_figures/fig_rq2_jacobian_largescale.py
echo "=== outputs ==="
ls -la results/data/all_in/_cache/_thesis_rq2_eps_prod_large*
