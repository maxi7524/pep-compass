#!/bin/bash
#SBATCH --job-name=rl_middle-1
#SBATCH --partition=common
#SBATCH --qos=kjurasz
#SBATCH --gres=gpu:rtx5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/rl_middle-1_%j.out
#SBATCH --error=logs/rl_middle-1_%j.err

PYTHON=~/pep-compass/.venv/bin/python

cd ~/pep-compass
mkdir -p logs results

echo "=================================================="
echo "Peptide: middle-1"
echo "Node   : $(hostname)"
echo "GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Python : $($PYTHON --version)"
echo "=================================================="

$PYTHON scripts/rl_peptide_optimizer.py run_all_peptides \
    --peptide_name="middle-1" \
    --n_episodes=100 \
    --max_steps=20 \
    --max_candidates=40 \
    --device=cuda \
    --output_dir=results

echo "Done: middle-1"