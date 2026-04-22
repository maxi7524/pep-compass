#!/bin/bash
#SBATCH --job-name=a2c_combo1
#SBATCH --partition=common
#SBATCH --qos=kjurasz
#SBATCH --gres=gpu:rtx5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/a2c_combo1_%j.out
#SBATCH --error=logs/a2c_combo1_%j.err

set -e
mkdir -p logs results

PYTHON=~/pep-compass/.venv/bin/python

echo "=========================================="
echo "EXTENDED A2C Training: combo1"
echo "Peptides: hydrodamin-2, mammuthusin-3"
echo "Host: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Started: $(date)"
echo "n_episodes=2000, max_steps=30"
echo "==========================================" 
echo "--- Starting peptide: hydrodamin-2 at $(date) ---"

$PYTHON scripts/rl_actor_critic_optimizer.py run_a2c_all_peptides \
    --peptide_name="hydrodamin-2" \
    --n_episodes=2000 \
    --max_steps=30 \
    --device=cuda \
    --output_dir=results

echo "--- Finished peptide: hydrodamin-2 at $(date) ---"

echo "--- Starting peptide: mammuthusin-3 at $(date) ---"

$PYTHON scripts/rl_actor_critic_optimizer.py run_a2c_all_peptides \
    --peptide_name="mammuthusin-3" \
    --n_episodes=2000 \
    --max_steps=30 \
    --device=cuda \
    --output_dir=results

echo "--- Finished peptide: mammuthusin-3 at $(date) ---"

echo "=========================================="
echo "ALL DONE: $(date)"
echo "=========================================="