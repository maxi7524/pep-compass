#!/bin/bash
#SBATCH --job-name=mpp-pogs
#SBATCH --partition=common
#SBATCH --qos=kjurasz
#SBATCH --gres=gpu:rtx5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=logs/mpp_pogs_%j.out
#SBATCH --error=logs/mpp_pogs_%j.err

# MUTANG+ whitened viability filter scored by the PoGS geodesic distance (lambda=0).
# Recomputes the whitened argmax/product tau-selection fresh (parent_geometry) and scores each
# candidate with d_pogs, mirroring _exp_mutangplus_eucl.py. Writes the tau-curve figure + summary.
#
# Submit:
#   sbatch scripts/mutangplus_pogs_gpu.sh
#   sbatch --export=ALL,MPE_LIMIT=5 scripts/mutangplus_pogs_gpu.sh   # smoke

set -euo pipefail
cd /home/kjurasz/pep-compass
mkdir -p logs results/rq5_figs

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found; refusing CPU fallback." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

PYTHON=/home/kjurasz/pep-compass/.venv/bin/python

export THESIS_FIG_DIR=/home/kjurasz/pep-compass/results/rq5_figs
export MPE_LIMIT=${MPE_LIMIT:-0}
export PGS_STEPS=${PGS_STEPS:-300}
export PGS_SEG=${PGS_SEG:-8}
export PYTHONUNBUFFERED=1

echo "MPE_LIMIT=${MPE_LIMIT} PGS_STEPS=${PGS_STEPS} PGS_SEG=${PGS_SEG} -> ${THESIS_FIG_DIR}"
time "${PYTHON}" analysis/scripts/thesis_figures/_exp_mutangplus_pogs.py
echo "=== outputs ==="
ls -la results/rq5_figs/rq5_mutangplus_pogs.pdf \
       results/data/all_in/_cache/_thesis_mutangplus_pogs_summary.csv 2>/dev/null || true
