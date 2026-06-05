#!/bin/bash
#SBATCH --job-name=rq7-score
#SBATCH --partition=common
#SBATCH --qos=kjurasz
#SBATCH --gres=gpu:rtx5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=logs/rq7_%j.out
#SBATCH --error=logs/rq7_%j.err

set -euo pipefail
cd /home/kjurasz/pep-compass
mkdir -p logs results/rq5_figs

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found; refusing CPU fallback."; exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

PYTHON=/home/kjurasz/pep-compass/.venv/bin/python
export THESIS_FIG_DIR=/home/kjurasz/pep-compass/results/rq5_figs
export PYTHONUNBUFFERED=1
echo "RQ7_LIMIT=${RQ7_LIMIT:-<full>}  RQ7_BENCH_ONLY=${RQ7_BENCH_ONLY:-0}"
time "${PYTHON}" analysis/scripts/thesis_figures/fig_rq7_scorings.py
echo "=== done ==="; ls -la results/rq5_figs/rq7_*.pdf 2>/dev/null
