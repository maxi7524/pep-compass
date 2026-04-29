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

# =========================
# CONFIG (single source of truth)
# =========================
WORKDIR=/home/kjurasz/pep-compass
VENV=$WORKDIR/.venv/bin/python

# =========================
# HARD FAIL SAFETY CHECKS
# =========================
if [ ! -d "$WORKDIR" ]; then
  echo "ERROR: WORKDIR not found: $WORKDIR"
  exit 1
fi

if [ ! -x "$VENV" ]; then
  echo "ERROR: Python venv not found: $VENV"
  exit 1
fi

cd "$WORKDIR"

# =========================
# LOGS SAFETY
# =========================
mkdir -p logs results/basic_eps_greedy_rl

# =========================
# GPU CHECK
# =========================
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found; refusing to run on CPU."
  exit 1
fi

echo "Host: $(hostname)"
echo "Date: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# =========================
# APEX MODELS CHECK
# =========================
APEX_DIR=$WORKDIR/src/pep_compass/models/apex
if [ ! -d "$APEX_DIR/APEX_pathogen_models" ] || [ ! -d "$APEX_DIR/Full_APEX_pathogen_models" ]; then
  echo "ERROR: APEX models not found in $APEX_DIR. Please ensure they are downloaded."
  exit 1
fi

# =========================
# RUN CONFIG
# =========================
N_AGENTS=5
N_EPOCHS=1500
MAX_STEPS=200

PIDS=()

echo "Starting $N_AGENTS agents..."

# =========================
# RUN AGENTS
# =========================
for W in $(seq 0 $((N_AGENTS-1))); do
  echo "Launching agent $W"

  "$VENV" scripts/rl_peptide_optimizer.py run_basic_epsilon_greedy \
    --n_epochs "$N_EPOCHS" \
    --max_steps "$MAX_STEPS" \
    --device cuda \
    --output_dir results/basic_eps_greedy_rl \
    --run_name "agent_$W" \
    --seed $((17 + W)) \
    --verbose false &

  PIDS+=($!)
done

# =========================
# WAIT + FAIL DETECTION
# =========================
FAIL=0

for PID in "${PIDS[@]}"; do
  if ! wait "$PID"; then
    FAIL=1
  fi
done

if [ "$FAIL" -ne 0 ]; then
  echo "ERROR: One or more agents failed"
  exit 1
fi

# =========================
# REPORT
# =========================
"$VENV" scripts/basic_eps_greedy_rl_report.py \
  --results_dir results/basic_eps_greedy_rl

echo "DONE: $(date)"