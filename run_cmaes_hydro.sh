#!/bin/bash
#
#SBATCH --job-name=cmaes_hydro
#SBATCH --partition=common
#SBATCH --qos=kjurasz
#SBATCH --gres=gpu:1
#SBATCH --time=180
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kjurasz@student.uw.edu.pl

timestamp=$(date +"%Y%m%d_%H%M%S")

# === OUTPUT DIRECTORY (on RAID) ===
output_dir="/raid/kjurasz/results/cmaes_hydrophobicity"
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
script_path="scripts/run_cmaes_optimization_hydro.py"
if [ ! -f "$script_path" ]; then
    echo "ERROR: Script not found at $script_path"
    exit 1
fi
echo "Script found: $script_path"

# === GPU INFO ===
echo "Checking GPU availability..."
nvidia-smi || echo "No GPU info available"
echo "CUDA version: $(nvcc --version 2>/dev/null || echo 'nvcc not found')"

# === SET ENVIRONMENT VARIABLES ===
echo "Setting up environment..."
export PYTHONPATH=/home/kjurasz/pep-compass/src
echo "PYTHONPATH set to: $PYTHONPATH"

# === CREATE RESULTS DIRECTORY ===
results_dir="$output_dir/optimization_results_$timestamp"
mkdir -p "$results_dir"
echo "Results will be saved to: $results_dir"

# Also ensure the CSV observer directory exists (this is what the script actually uses)
csv_results_dir="$results_dir/cma_es_hydrophobicity"
mkdir -p "$csv_results_dir"
echo "CSV results directory: $csv_results_dir"

# === MODIFY SCRIPT TO USE RAID OUTPUT ===
# Create a temporary script with updated output path
temp_script="/tmp/run_cmaes_optimization_hydro_$timestamp.py"
sed "s|\"./results/cma_es_hydrophobicity\"|\"$csv_results_dir\"|g" "$script_path" > "$temp_script"
echo "Created temporary script with updated output path: $temp_script"
echo "Script will save CSV files to: $csv_results_dir"

# === RUN OPTIMIZATION ===
echo "=== Starting CMA-ES hydrophobicity optimization at $(date) ==="

python3 "$temp_script"

exit_code=$?
echo "=== Script finished with exit code: $exit_code ==="

if [ $exit_code -eq 0 ]; then
    echo "SUCCESS: Optimization completed"
    if [ -d "$results_dir" ]; then
        echo "Results directory exists:"
        ls -lah "$results_dir"
        echo "Number of result files:"
        find "$results_dir" -name "*.csv" | wc -l
        echo "Sample files:"
        find "$results_dir" -name "*.csv" | head -3
    fi
else
    echo "ERROR: Script failed with exit code $exit_code"
fi

# === CLEANUP ===
echo "Cleaning up temporary files..."
rm -f "$temp_script"

echo "=== Job completed at $(date) ==="
echo "Log file: $log_file"
echo "Results directory: $results_dir"
echo "Output directory: $output_dir"