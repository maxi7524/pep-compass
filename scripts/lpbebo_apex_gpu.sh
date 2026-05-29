#!/bin/bash
#SBATCH --job-name=lpbebo-apex
#SBATCH --partition=common
#SBATCH --qos=kjurasz
#SBATCH --gres=gpu:rtx5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=logs/lpbebo_apex_%j.out
#SBATCH --error=logs/lpbebo_apex_%j.err

set -euo pipefail

cd /home/kjurasz/pep-compass
mkdir -p logs results/lpbebo

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found; refusing CPU fallback."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

PYTHON=/home/kjurasz/pep-compass/.venv/bin/python

"${PYTHON}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; refusing CPU fallback.")
print("CUDA devices:", torch.cuda.device_count())
PY

"${PYTHON}" scripts/run_lpbebo_optimization_apex.py
