#!/bin/bash
#SBATCH --job-name=geo-bench
#SBATCH --partition=common
#SBATCH --qos=kjurasz
#SBATCH --gres=gpu:rtx5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=logs/geo_bench_%j.out
#SBATCH --error=logs/geo_bench_%j.err

set -euo pipefail

cd /home/kjurasz/pep-compass
mkdir -p logs results

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found; refusing CPU fallback."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

PYTHON=/home/kjurasz/pep-compass/.venv/bin/python

"${PYTHON}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable; refusing CPU fallback.")
print("CUDA devices:", torch.cuda.device_count(), torch.cuda.get_device_name(0))
PY

echo "=== geometry self-test (analytic ground truth) ==="
"${PYTHON}" analysis/scripts/thesis_figures/_geometry_selftest.py

echo "=== geodesic distance benchmark (GPU) ==="
"${PYTHON}" scripts/bench_geodesic_distance.py --device cuda
