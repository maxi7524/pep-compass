#!/bin/bash
#SBATCH --job-name=a2c_jurand-4
#SBATCH --partition=common
#SBATCH --qos=kjurasz
#SBATCH --gres=gpu:rtx5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/a2c_jurand-4_%j.out
#SBATCH --error=logs/a2c_jurand-4_%j.err

set -e

mkdir -p logs

PYTHON=~/pep-compass/.venv/bin/python

echo "=========================================="
echo "A2C Actor-Critic Optimizer: jurand-4"
echo "Host: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Started: $(date)"
echo "=========================================="

$PYTHON scripts/rl_actor_critic_optimizer.py run_a2c_all_peptides \
    --peptide_name="jurand-4" \
    --n_episodes=100 \
    --max_steps=20 \
    --device=cuda \
    --output_dir=results

echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
