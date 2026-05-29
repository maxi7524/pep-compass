#!/bin/bash
#SBATCH --job-name=eps-promising
#SBATCH --partition=common
#SBATCH --qos=kjurasz
#SBATCH --gres=gpu:rtx5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eps_promising_%j.out
#SBATCH --error=logs/eps_promising_%j.err

set -euo pipefail

cd /home/kjurasz/pep-compass
mkdir -p logs basic_eps_greedy_rl/inputs results/basic_eps_greedy_rl_promising_start

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found; refusing CPU fallback."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

PYTHON=/home/kjurasz/pep-compass/.venv/bin/python
PEPTIDE_LIST=basic_eps_greedy_rl/inputs/sampled_500_peptides.txt

"${PYTHON}" scripts/sample_peptides_from_csvs.py \
  --root /home/kjurasz/pep-compass \
  --dataset_subdir results/mutants/mutants \
  --sample_size 500 \
  --max_len 25 \
  --seed 2026 \
  --output_file "${PEPTIDE_LIST}" \
  --meta_file basic_eps_greedy_rl/inputs/sampled_500_peptides_meta.json

# Promising start strategy: coverage-first curriculum through a broad peptide pool.
"${PYTHON}" scripts/rl_peptide_optimizer.py run_five_agents \
  --n_agents 6 \
  --n_epochs 1500 \
  --max_steps 200 \
  --device cuda \
  --output_dir results/basic_eps_greedy_rl_promising_start \
  --start_selection cycle \
  --start_peptides_file "${PEPTIDE_LIST}" \
  --verbose false

"${PYTHON}" scripts/basic_eps_greedy_rl_report.py \
  --results_dir results/basic_eps_greedy_rl_promising_start
