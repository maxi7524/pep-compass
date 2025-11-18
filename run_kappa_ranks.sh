#!/bin/bash
#
#SBATCH --job-name=kappa_ranks
#SBATCH --partition=common
#SBATCH --qos=kjurasz
#SBATCH --gres=gpu:1
#SBATCH --time=10
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kjurasz@student.uw.edu.pl

timestamp=$(date +"%Y%m%d_%H%M%S")

# === OUTPUT DIRECTORY (on RAID) ===
output_dir="/raid/kjurasz/results/kappa_ranks"
mkdir -p "$output_dir"

# Log file path
log_file="$output_dir/logs_$timestamp.txt"

# Redirect stdout + stderr before any output
exec > "$log_file" 2>&1

echo "=== SLURM Job Started at $(date) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "User: $USER"
echo "Timestamp: $timestamp"
echo "Output directory: $output_dir"
echo "Log file: $log_file"

# === ACTIVATE VIRTUAL ENV ===
echo "Activating virtual environment..."
source /home/kjurasz/pep-compass/.venv/bin/activate
echo "Virtual environment activated: $VIRTUAL_ENV"

# === CHANGE TO PROJECT DIRECTORY ===
echo "Changing to project directory..."
cd /home/kjurasz/pep-compass
echo "Current directory: $(pwd)"

# === CHECK SCRIPT EXISTS ===
script_path="analysis/notebooks/dev/calculate_ranks.py"
if [ ! -f "$script_path" ]; then
    echo "ERROR: Script not found at $script_path"
    exit 1
fi
echo "Script found: $script_path"

# === GPU INFO ===
echo "Checking GPU availability..."
nvidia-smi || echo "No GPU info available"
echo "CUDA version: $(nvcc --version 2>/dev/null || echo 'nvcc not found')"

# === RUN CALCULATION ===
output_file="$output_dir/ranks_multi_$timestamp.csv"
echo "=== Starting rank calculation at $(date) ==="
echo "Output file: $output_file"

python3 "$script_path" \
    -i "ACDEFGH,KLMNPQR,STVWY,DEFGHIK,EFGHIKL" \
    -k "0.01,0.001,0.0001,0.00001" \
    -d "cuda" \
    -o "$output_file"

exit_code=$?
echo "=== Script finished with exit code: $exit_code ==="

if [ $exit_code -eq 0 ]; then
    echo "SUCCESS: Results saved to $output_file"
    if [ -f "$output_file" ]; then
        echo "File exists:"
        ls -lh "$output_file"
        echo "First few lines:"
        head -5 "$output_file"
    fi
else
    echo "ERROR: Script failed with exit code $exit_code"
fi

echo "=== Job completed at $(date) ==="
echo "Log file: $log_file"
echo "Results directory: $output_dir"
