#!/bin/bash
#SBATCH --job-name=rq5-geo
#SBATCH --partition=common
#SBATCH --qos=kjurasz
#SBATCH --gres=gpu:rtx5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=logs/rq5_geo_%j.out
#SBATCH --error=logs/rq5_geo_%j.err

set -euo pipefail
cd /home/kjurasz/pep-compass
mkdir -p logs results/rq5_figs

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found; refusing CPU fallback."; exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

PYTHON=/home/kjurasz/pep-compass/.venv/bin/python

# Knobs (override at submit with --export):
#   RQ5_FEAS  = geo | maha | eucl       (distance feeding the feasibility analysis)
#   RQ5_LIMIT = N sample peptides       (small for a gate-check, unset = full ~500)
#   RQ5_GEO_K = kNN k for the geodesic graph
export THESIS_FIG_DIR=/home/kjurasz/pep-compass/results/rq5_figs
export RQ5_FEAS=${RQ5_FEAS:-geo}
export PYTHONUNBUFFERED=1

echo "RQ5_FEAS=${RQ5_FEAS}  RQ5_LIMIT=${RQ5_LIMIT:-<full>}  RQ5_GEO_K=${RQ5_GEO_K:-12}  -> ${THESIS_FIG_DIR}"
time "${PYTHON}" analysis/scripts/thesis_figures/fig_rq56_potentials.py
echo "=== figures written ==="
ls -la results/rq5_figs/
