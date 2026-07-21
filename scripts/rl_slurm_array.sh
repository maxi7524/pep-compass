#!/bin/bash
#SBATCH --job-name=rl_pep_array
#SBATCH --partition=common
#SBATCH --qos=kjurasz
#SBATCH --gres=gpu:rtx5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/rl_array_%A_%a.out
#SBATCH --error=logs/rl_array_%A_%a.err
#SBATCH --array=0-5

# ── peptide names matching SEED_PEPTIDES keys in rl_peptide_optimizer.py ──
PEPTIDE_NAMES=(
    "middle-1"
    "jurand-4"
    "jurand-2"
    "jurand-7"
    "mammuthusin-3"
    "hydrodamin-2"
)

PEPTIDE_NAME=${PEPTIDE_NAMES[$SLURM_ARRAY_TASK_ID]}

cd ~/pep-compass
mkdir -p logs results

echo "=================================================="
echo "Array task  : $SLURM_ARRAY_TASK_ID"
echo "Peptide name: $PEPTIDE_NAME"
echo "Node        : $(hostname)"
echo "GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "=================================================="

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
source ~/.bashrc 2>/dev/null || true

# Run smoke test first
echo "--- smoke test ---"
python scripts/rl_peptide_optimizer.py test_components --device=cuda

echo "--- optimisation ---"
python scripts/rl_peptide_optimizer.py run_all_peptides \
    --peptide_name="$PEPTIDE_NAME" \
    --n_episodes=100 \
    --max_steps=20 \
    --max_candidates=40 \
    --device=cuda \
    --output_dir=results \
    --lr=0.001 \
    --gamma=0.99 \
    --epsilon_start=1.0 \
    --epsilon_end=0.05 \
    --batch_size=32 \
    --buffer_capacity=10000 \
    --target_update_freq=50

echo "Done: $PEPTIDE_NAME"