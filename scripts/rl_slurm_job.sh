#!/bin/bash
#SBATCH --job-name=rl_peptide
#SBATCH --partition=common
#SBATCH --qos=kjurasz
#SBATCH --gres=gpu:rtx5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/home/kjurasz/pep-compass/results/rl_%j.log
#SBATCH --error=/home/kjurasz/pep-compass/results/rl_%j.err

set -euo pipefail

cd /home/kjurasz/pep-compass
mkdir -p results

/usr/bin/nvidia-smi -L

echo "=== Starting RL test_components ==="
.venv/bin/python scripts/rl_peptide_optimizer.py test_components

echo "=== Starting RL run ==="
.venv/bin/python scripts/rl_peptide_optimizer.py run_rl_optimization \
    --n_episodes=50 \
    --max_steps=20 \
    --max_candidates=40 \
    --buffer_capacity=2000 \
    --batch_size=32 \
    --output_dir=results \
    --device=cuda

echo "=== Done ==="