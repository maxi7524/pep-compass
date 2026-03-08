#!/bin/bash
#SBATCH --job-name=td3_hydrodamin-2
#SBATCH --partition=common
#SBATCH --qos=kjurasz
#SBATCH --gres=gpu:rtx5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/td3_hydrodamin-2_%j.out
#SBATCH --error=logs/td3_hydrodamin-2_%j.err

set -e

mkdir -p logs

PYTHON=~/pep-compass/.venv/bin/python

echo "=========================================="
echo "TD3 Continuous Optimizer: hydrodamin-2"
echo "Host: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Started: $(date)"
echo "=========================================="

$PYTHON scripts/rl_continuous_optimizer.py run_td3_all_peptides \
    --peptide_name="hydrodamin-2" \
    --n_episodes=100 \
    --max_steps=20 \
    --device=cuda \
    --output_dir=results

echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
