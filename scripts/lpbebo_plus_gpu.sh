#!/bin/bash
#SBATCH --job-name=lpbebo-plus
#SBATCH --partition=common
#SBATCH --qos=kjurasz
#SBATCH --gres=gpu:rtx5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=logs/lpbebo_plus_%j.out
#SBATCH --error=logs/lpbebo_plus_%j.err

set -euo pipefail
cd /home/kjurasz/pep-compass
mkdir -p logs results/lpbebo_plus

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found; refusing CPU fallback."; exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

PYTHON=/home/kjurasz/pep-compass/.venv/bin/python
export PYTHONUNBUFFERED=1
# knobs (override at submit): LPBEBO_PEPTIDE, LPBEBO_N_TRAJ, LPBEBO_BUDGET, LPBEBO_OUT
echo "peptide=${LPBEBO_PEPTIDE:-default} n_traj=${LPBEBO_N_TRAJ:-10} budget=${LPBEBO_BUDGET:-1400}"
time "${PYTHON}" scripts/lpbebo_plus.py
echo "=== done ==="; ls -la results/lpbebo_plus/ 2>/dev/null | tail
