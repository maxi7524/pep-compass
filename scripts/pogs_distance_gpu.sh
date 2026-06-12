#!/bin/bash
#SBATCH --job-name=pogs-dist
#SBATCH --partition=common
#SBATCH --qos=kjurasz
#SBATCH --gres=gpu:rtx5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=logs/pogs_dist_%j.out
#SBATCH --error=logs/pogs_dist_%j.err

# Add the PoGS (Potential-minimizing Geodesic Search, lambda=0) distance column to the
# RQ3/feasibility candidate caches. PoGS path energy is ADAM-minimised per candidate; the
# reported distance is the ambient chord length of the optimised path (no metric G).
#
# Submit:
#   sbatch scripts/pogs_distance_gpu.sh
#   # Quick smoke (5 parents per cache):
#   sbatch --export=ALL,PGS_LIMIT=5 scripts/pogs_distance_gpu.sh
#   # Only one cache:
#   sbatch --export=ALL,PGS_WHICH=rq5 scripts/pogs_distance_gpu.sh
#
# Outputs land in results/data/all_in/_cache/:
#   _thesis_rq5_distances_pogs.parquet   (385 peptides; has score_A_onehot/diff for TANDEM-A)
#   _thesis_mutangplus_pogs.parquet      (845 peptides; for MUTANG+ tau-curve)

set -euo pipefail
cd /home/kjurasz/pep-compass
mkdir -p logs

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found; refusing CPU fallback." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

PYTHON=/home/kjurasz/pep-compass/.venv/bin/python

export PGS_WHICH=${PGS_WHICH:-both}
export PGS_LIMIT=${PGS_LIMIT:-0}
export PGS_STEPS=${PGS_STEPS:-500}
export PGS_SEG=${PGS_SEG:-8}
export PGS_MU=${PGS_MU:-1e-2}
export PGS_LR=${PGS_LR:-1e-3}
export PGS_CHUNK=${PGS_CHUNK:-1024}
export PYTHONUNBUFFERED=1

echo "PGS_WHICH=${PGS_WHICH} PGS_LIMIT=${PGS_LIMIT} PGS_STEPS=${PGS_STEPS} PGS_SEG=${PGS_SEG} PGS_MU=${PGS_MU} PGS_LR=${PGS_LR}"
time "${PYTHON}" analysis/scripts/thesis_figures/_exp_pogs_distance_cache.py
echo "=== outputs ==="
ls -la results/data/all_in/_cache/_thesis_rq5_distances_pogs.parquet \
       results/data/all_in/_cache/_thesis_mutangplus_pogs.parquet 2>/dev/null || true
