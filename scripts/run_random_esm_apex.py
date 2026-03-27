from datetime import datetime
import time
import sys
import os
import warnings
import logging

from pep_compass.optimization.black_box.apex_black_box import APEXBlackBox

# Comprehensive warning suppression
warnings.filterwarnings('ignore')
os.environ['RDKIT_QUIET'] = '1'
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow warnings
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

# Suppress all logging
logging.getLogger().setLevel(logging.ERROR)
for logger_name in ['rdkit', 'tensorflow', 'torch', 'transformers', 'pytorch']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# Also set RDKit logger to suppress messages
try:
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    pass

# Redirect stderr to suppress various C++ warnings (like NNPACK)
import io
from contextlib import redirect_stderr

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from pep_compass.optimization.baselines.random_mutation import RandomMutationOptimizer
from pep_compass.optimization.black_box.csv_observer import CSVObserver

DEVICE = "cuda:0"
OUTPUT_PATH = "./results/random_mutation"
EVALUATION_BUDGET = 1400

# --- ESM CHANGE: ESM2 filter configuration (lightest model + requested threshold) ---
ESM_MODEL_NAME = "esm2_t6_8M_UR50D"
ESM_DEVICE = "cpu"
ESM_PPL_THRESHOLD = -0.5
ESM_MAX_RESAMPLING_ATTEMPTS = 200

black_box = APEXBlackBox(
    mic_aggregate="mean",
    mic_bacteria=[1, 2, 3],
    device=DEVICE,
)

observer = CSVObserver(maximize = black_box.maximize)
black_box.set_observer(observer)

optimizer = RandomMutationOptimizer(
    black_box=black_box,
    # --- ESM CHANGE: enable ESM-based mutation rejection ---
    esm_model_name=ESM_MODEL_NAME,
    esm_ppl_threshold=ESM_PPL_THRESHOLD,
    esm_device=ESM_DEVICE,
    esm_max_resampling_attempts=ESM_MAX_RESAMPLING_ATTEMPTS,
)

proteins = {
    # "middle-1": ("FLYKWWIRIGRLKL", 5),
    "KY14": ("KYCRRFRWLTFRWL", 5),  # --- ESM CHANGE: explicit KY14 alias ---
    # "jurand-4": ("KYCRRFRWLTFRWL", 5),
    # "jurand-2": ("KFRNRHRWKFKLIFRN", 5),
    # "jurand-7": ("KKYWLIRKWIRLWFLT", 5),
    # "mammuthusin-3": ("KTLKIIRLLF", 5),
    # "hydrodamin-2": ("RMARNLVRYVQGLKKKKVI", 5),
}

for name, (sequence, num) in proteins.items():
    for i in range(num):
        print(f"Starting optimization for {name}, iteration {i+1}")
        rng_seed = int(time.time())  # Create a unique rng_seed for each iteration
        observer.initialize_observer(
            black_box.get_black_box_info(),
            {
                "experiment_id": f"{rng_seed}_{sequence}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "experiment_path": OUTPUT_PATH,
            },
            rng_seed,
        )
        # Suppress stderr during optimization to hide RDKit and NNPACK warnings
        devnull = io.StringIO()
        old_stderr = sys.stderr
        try:
            sys.stderr = devnull
            with redirect_stderr(devnull):
                optimizer.optimize(
                    evaluation_budget=EVALUATION_BUDGET, starting_point=sequence, rng_seed=rng_seed
                )
        finally:
            sys.stderr = old_stderr

