#!/bin/bash
#
#SBATCH --job-name=battleamp_gpu_test
#SBATCH --partition=common
#SBATCH --qos=kjurasz
#SBATCH --gres=gpu:1
#SBATCH --time=10
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kjurasz@student.uw.edu.pl

timestamp=$(date +"%Y%m%d_%H%M%S")

# === OUTPUT DIRECTORY (local) ===
output_dir="/home/kjurasz/pep-compass/results/battleamp_gpu_test"
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
source /home/kjurasz/pep-compass/.env/bin/activate
echo "Virtual environment activated: $VIRTUAL_ENV"

# === CHANGE TO PROJECT DIRECTORY ===
echo "Changing to project directory..."
cd /home/kjurasz/pep-compass
echo "Current directory: $(pwd)"

# === CHECK BATTLEAMP PREDICTOR EXISTS ===
predictor_path="src/pep_compass/models/battleamp/BattleAMPPredictor.py"
if [ ! -f "$predictor_path" ]; then
    echo "ERROR: BattleAMP predictor not found at $predictor_path"
    exit 1
fi
echo "BattleAMP predictor found: $predictor_path"

# === GPU INFO ===
echo "Checking GPU availability..."
nvidia-smi || echo "No GPU info available"
echo "CUDA version: $(nvcc --version 2>/dev/null || echo 'nvcc not found')"

# === CHECK PYTHON ENVIRONMENT ===
echo "Python version: $(python3 --version)"
echo "TensorFlow version: $(python3 -c "import tensorflow as tf; print(tf.__version__)" 2>/dev/null || echo 'TensorFlow not found')"
echo "PyTorch version: $(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo 'PyTorch not found')"

# === RUN BATTLEAMP GPU TEST ===
output_file="$output_dir/battleamp_gpu_results_$timestamp.txt"
echo "=== Starting BattleAMP GPU test at $(date) ==="
echo "Output file: $output_file"

# Create and run BattleAMP test script
echo "Creating BattleAMP test script..."
cat > "$output_dir/battleamp_test_$timestamp.py" << 'EOF'
import sys
import os
# Add both the src directory and current directory to Python path
sys.path.insert(0, '/home/kjurasz/pep-compass/src')
sys.path.insert(0, '/home/kjurasz/pep-compass')

import tensorflow as tf
import numpy as np
import time

print("=== BattleAMP BlackBox GPU Test ===")
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
print(f"GPU available: {tf.test.is_gpu_available()}")

# Test sequences as arrays (like the black box expects)
test_sequences = [
    list("KLLLKLLKKLLKLLK"),
    list("FLPIIAKLLGLL"), 
    list("WLGHFTVRK")
]
test_array = np.array(test_sequences, dtype=object)

try:
    print("\n--- Testing BattleAMP BlackBox ---")
    # Test BattleAMP predictor directly first (avoids PyTorch import conflict)
    from pep_compass.models.battleamp.BattleAMPPredictor import PredictorBattleAMP
    
    print("\n1. Testing BattleAMP Predictor directly:")
    pred_cpu = PredictorBattleAMP(device="cpu")
    pred_gpu = PredictorBattleAMP(device="cuda")
    
    test_seqs = ["KLLLKLLKKLLKLLK", "FLPIIAKLLGLL", "WLGHFTVRK"]
    
    start_time = time.time()
    cpu_results = pred_cpu.predict(test_seqs)
    cpu_time = time.time() - start_time
    print(f"CPU Time: {cpu_time:.3f}s, Results: {cpu_results.flatten()}")
    
    start_time = time.time() 
    gpu_results = pred_gpu.predict(test_seqs)
    gpu_time = time.time() - start_time
    print(f"GPU Time: {gpu_time:.3f}s, Results: {gpu_results.flatten()}")
    
    if cpu_time > 0 and gpu_time > 0:
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")
    
    print("\n2. Testing BattleAMP BlackBox (if no PyTorch conflicts):")
    from pep_compass.optimization.black_box.battleamp_black_box import BattleAMPBlackBox
    
    print("\n1. CPU BlackBox Test:")
    bb_cpu = BattleAMPBlackBox(device="cpu")
    start_time = time.time()
    cpu_results = bb_cpu._black_box(test_array)
    cpu_time = time.time() - start_time
    print(f"CPU Time: {cpu_time:.3f}s")
    print(f"CPU Results shape: {cpu_results.shape}")
    print(f"CPU Results: {cpu_results.flatten()}")
    
    print("\n2. GPU BlackBox Test:")
    bb_gpu = BattleAMPBlackBox(device="cuda")
    start_time = time.time()
    gpu_results = bb_gpu._black_box(test_array)
    gpu_time = time.time() - start_time
    print(f"GPU Time: {gpu_time:.3f}s")
    print(f"GPU Results shape: {gpu_results.shape}")
    print(f"GPU Results: {gpu_results.flatten()}")
    
    print(f"\n3. Performance Comparison:")
    if cpu_time > 0 and gpu_time > 0:
        speedup = cpu_time / gpu_time
        print(f"Speedup (CPU/GPU): {speedup:.2f}x")
        if speedup > 1.1:
            print("✓ GPU provides speedup!")
        elif speedup < 0.9:
            print("⚠ GPU slower (overhead for small batches)")
        else:
            print("≈ Similar performance")
    
    # Check result consistency
    diff = np.abs(cpu_results - gpu_results).max()
    print(f"Max difference between CPU/GPU: {diff:.6f}")
    if diff < 1e-5:
        print("✓ Results are consistent between CPU and GPU")
    else:
        print("⚠ Results differ between CPU and GPU")
    
    print(f"\n4. BlackBox Info:")
    print(bb_gpu.get_black_box_info())
    
    print("\nSUCCESS: BattleAMP BlackBox GPU test completed")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
EOF

echo "Running BattleAMP test script..."
python3 "$output_dir/battleamp_test_$timestamp.py" > "$output_file" 2>&1
exit_code=$?

echo "=== Script finished with exit code: $exit_code ==="

if [ $exit_code -eq 0 ]; then
    echo "SUCCESS: Results saved to $output_file"
    if [ -f "$output_file" ]; then
        echo "File exists:"
        ls -lh "$output_file"
        echo "First few lines:"
        head -10 "$output_file"
        echo ""
        echo "Last few lines:"
        tail -10 "$output_file"
    fi
else
    echo "ERROR: Script failed with exit code $exit_code"
    if [ -f "$output_file" ]; then
        echo "Error output:"
        tail -20 "$output_file"
    fi
fi

echo "=== Job completed at $(date) ==="
echo "Log file: $log_file"
echo "Results directory: $output_dir"
echo "Available files:"
ls -la "$output_dir/"