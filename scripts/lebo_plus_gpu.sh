#!/bin/bash
#SBATCH --job-name=lebo-plus
#SBATCH --partition=common
#SBATCH --qos=kjurasz
#SBATCH --gres=gpu:rtx5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=logs/lebo_plus_%j.out
#SBATCH --error=logs/lebo_plus_%j.err

set -euo pipefail
cd /home/kjurasz/pep-compass
mkdir -p logs results/lebo_plus

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found; refusing CPU fallback."; exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

PYTHON=/home/kjurasz/pep-compass/.venv/bin/python
export PYTHONUNBUFFERED=1

# knobs (override at submit):
#   LEBO_PLUS_APEX=1   -> 6-peptide APEX benchmark (29 trajectories total)
#   LEBO_PLUS_BUDGET   -> evaluation budget per trajectory (default 1400; small for smoke)
#   LEBO_PLUS_TAU      -> MUTANG+ whitened-similarity threshold (default 0.15)
#   LEBO_PLUS_OUT      -> output dir (default ./results/lebo_plus)
echo "apex=${LEBO_PLUS_APEX:-0} budget=${LEBO_PLUS_BUDGET:-1400} tau=${LEBO_PLUS_TAU:-0.15}"
time "${PYTHON}" scripts/lebo_plus.py
echo "=== done ==="
ls -la results/lebo_plus/ 2>/dev/null | tail
