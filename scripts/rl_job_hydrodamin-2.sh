#!/bin/bash
#SBATCH --job-name=rl_hydrodamin-2
#SBATCH --partition=common
#SBATCH --qos=kjurasz
#SBATCH --gres=gpu:rtx5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/rl_hydrodamin-2_%j.out
#SBATCH --error=logs/rl_hydrodamin-2_%j.err

PYTHON=~/pep-compass/.venv/bin/python

cd ~/pep-compass
mkdir -p logs results

echo "=================================================="
echo "Peptide: hydrodamin-2"
echo "Node   : $(hostname)"
echo "GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Python : $($PYTHON --version)"
echo "=================================================="

$PYTHON scripts/rl_peptide_optimizer.py run_all_peptides \
    --peptide_name="hydrodamin-2" \
    --n_episodes=100 \
    --max_steps=20 \
    --max_candidates=40 \
    --device=cuda \
    --output_dir=results

echo "Done: hydrodamin-2"