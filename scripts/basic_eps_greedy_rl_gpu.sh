#!/bin/bash
#SBATCH --job-name=basic_eps_rl
#SBATCH --partition=common
#SBATCH --qos=kjurasz
#SBATCH --gres=gpu:rtx5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/basic_eps_greedy_rl_%j.out
#SBATCH --error=logs/basic_eps_greedy_rl_%j.err

set -euo pipefail

cd /home/kjurasz/pep-compass.worktrees/rl_trials
mkdir -p logs results/basic_eps_greedy_rl

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found; refusing to run on CPU."
  exit 1
fi

echo "Host: $(hostname)"
echo "Date: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

APEX_DIR="src/pep_compass/models/apex"
if [ ! -d "${APEX_DIR}/APEX_pathogen_models" ]; then
  cp -r /home/kjurasz/pep-compass/src/pep_compass/models/apex/APEX_pathogen_models "${APEX_DIR}/"
fi
if [ ! -d "${APEX_DIR}/Full_APEX_pathogen_models" ]; then
  cp -r /home/kjurasz/pep-compass/src/pep_compass/models/apex/Full_APEX_pathogen_models "${APEX_DIR}/"
fi

PYTHON=/home/kjurasz/pep-compass/.venv/bin/python

"${PYTHON}" scripts/rl_peptide_optimizer.py run_five_agents \
  --n_agents 5 \
  --n_epochs 1500 \
  --max_steps 200 \
  --device cuda \
  --output_dir results/basic_eps_greedy_rl \
  --start_selection random \
  --verbose false

"${PYTHON}" scripts/basic_eps_greedy_rl_report.py \
  --results_dir results/basic_eps_greedy_rl

echo "Done: $(date)"
