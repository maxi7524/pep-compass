#!/bin/bash
#SBATCH --job-name=rm_apex_ky14
#SBATCH --partition=common
#SBATCH --qos=kjurasz
#SBATCH --gres=gpu:rtx5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/rm_apex_ky14_%j.out
#SBATCH --error=logs/rm_apex_ky14_%j.err

set -e
mkdir -p logs results/random_mutation

PYTHON=~/pep-compass/.venv/bin/python

echo "=========================================="
echo "Random Mutation APEX (KY14)"
echo "Host: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Started: $(date)"
echo "=========================================="

$PYTHON scripts/run_random_mutation_optimization.py \
    --protein-key KY14 \
    --repeats 1 \
    --device cuda:0 \
    --esm-device cuda:0

echo "Finished: $(date)"
