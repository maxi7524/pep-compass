#!/bin/bash
#SBATCH --job-name=eps500x10
#SBATCH --partition=common
#SBATCH --qos=kjurasz
#SBATCH --gres=gpu:rtx5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eps500x10_%j.out
#SBATCH --error=logs/eps500x10_%j.err

set -euo pipefail

cd /home/kjurasz/pep-compass
mkdir -p logs basic_eps_greedy_rl/inputs results/basic_eps_greedy_rl_500x10

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found; refusing CPU fallback."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

PYTHON=/home/kjurasz/pep-compass/.venv/bin/python
PEPTIDE_LIST=basic_eps_greedy_rl/inputs/sampled_500_peptides.txt
META_JSON=basic_eps_greedy_rl/inputs/sampled_500_peptides_meta.json

"${PYTHON}" scripts/sample_peptides_from_csvs.py \
  --root /home/kjurasz/pep-compass \
  --dataset_subdir results/mutants/mutants \
  --sample_size 500 \
  --max_len 25 \
  --seed 2026 \
  --output_file "${PEPTIDE_LIST}" \
  --meta_file "${META_JSON}"

if [ ! -s "${PEPTIDE_LIST}" ]; then
  echo "ERROR: peptide list was not created: ${PEPTIDE_LIST}"
  exit 1
fi

PIDS=()
for W in 0 1 2 3 4 5; do
  "${PYTHON}" scripts/basic_eps_greedy_dataset_runner.py \
    --peptides_file "${PEPTIDE_LIST}" \
    --repeats_per_peptide 10 \
    --num_workers 6 \
    --worker_id "${W}" \
    --n_epochs 1500 \
    --max_steps 200 \
    --device cuda \
    --output_dir results/basic_eps_greedy_rl_500x10 \
    --start_selection random &
  PIDS+=($!)
done

FAIL=0
for PID in "${PIDS[@]}"; do
  if ! wait "${PID}"; then
    FAIL=1
  fi
done

if [ "${FAIL}" -ne 0 ]; then
  echo "ERROR: One or more worker processes failed."
  exit 1
fi
