from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import time
import warnings
from contextlib import redirect_stderr
from datetime import datetime

# Comprehensive warning suppression
warnings.filterwarnings("ignore")
os.environ["RDKIT_QUIET"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

logging.getLogger().setLevel(logging.ERROR)
for logger_name in ["rdkit", "tensorflow", "torch", "transformers", "pytorch"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

try:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except ImportError:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from pep_compass.optimization.baselines.random_mutation import RandomMutationOptimizer
from pep_compass.optimization.black_box.apex_black_box import APEXBlackBox
from pep_compass.optimization.black_box.csv_observer import CSVObserver

DEFAULT_PROTEINS = {
    "KY14": "KYCRRFRWLTFRWL",
    "middle-1": "FLYKWWIRIGRLKL",
    "jurand-4": "KYCRRFRWLTFRWL",
    "jurand-2": "KFRNRHRWKFKLIFRN",
    "jurand-7": "KKYWLIRKWIRLWFLT",
    "mammuthusin-3": "KTLKIIRLLF",
    "hydrodamin-2": "RMARNLVRYVQGLKKKKVI",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run APEX random mutation optimization with optional ESM2 PLL filtering."
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device for APEX black-box model.",
    )
    parser.add_argument(
        "--mic-aggregate",
        default="mean",
        choices=["mean", "min", "max"],
        help="How to aggregate APEX pathogen MIC predictions.",
    )
    parser.add_argument(
        "--mic-bacteria",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Bacteria IDs used by APEX black box.",
    )
    parser.add_argument(
        "--esm-device",
        default="cpu",
        help="Device for ESM scoring model.",
    )
    parser.add_argument(
        "--esm-model-name",
        default="esm2_t6_8M_UR50D",
        help="ESM2 model name for PLL filtering.",
    )
    parser.add_argument(
        "--esm-ppl-threshold",
        type=float,
        default=-0.5,
        help="Reject mutation candidates with PLL below this threshold.",
    )
    parser.add_argument(
        "--disable-esm-filter",
        action="store_true",
        help="Disable ESM PLL filtering.",
    )
    parser.add_argument(
        "--esm-max-resampling-attempts",
        type=int,
        default=200,
        help="Max mutation retries per step to satisfy ESM threshold.",
    )
    parser.add_argument(
        "--evaluation-budget",
        type=int,
        default=1400,
        help="Number of random mutation steps.",
    )
    parser.add_argument(
        "--protein-key",
        default="KY14",
        help="Protein name for experiment metadata.",
    )
    parser.add_argument(
        "--starting-sequence",
        default=None,
        help="Starting peptide sequence. Required for unknown protein keys.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of repeated runs.",
    )
    parser.add_argument(
        "--output-path",
        default="./results/random_mutation",
        help="Base path for CSV observer output.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    black_box = APEXBlackBox(
        mic_aggregate=args.mic_aggregate,
        mic_bacteria=args.mic_bacteria,
        device=args.device,
    )
    observer = CSVObserver(maximize=black_box.maximize)
    black_box.set_observer(observer)

    default_sequence = DEFAULT_PROTEINS.get(args.protein_key)
    sequence = args.starting_sequence or default_sequence
    if not sequence:
        available = ", ".join(sorted(DEFAULT_PROTEINS))
        raise ValueError(
            f"No sequence configured for protein '{args.protein_key}'. "
            f"Provide --starting-sequence. Available preset keys: {available}"
        )

    optimizer = RandomMutationOptimizer(
        black_box=black_box,
        esm_model_name=None if args.disable_esm_filter else args.esm_model_name,
        esm_ppl_threshold=args.esm_ppl_threshold,
        esm_device=args.esm_device,
        esm_max_resampling_attempts=args.esm_max_resampling_attempts,
    )

    for repeat_idx in range(args.repeats):
        print(f"Starting optimization for {args.protein_key}, iteration {repeat_idx + 1}")
        rng_seed = int(time.time())

        observer.initialize_observer(
            black_box.get_black_box_info(),
            {
                "experiment_id": f"{args.protein_key}_{rng_seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "experiment_path": args.output_path,
            },
            rng_seed,
        )

        devnull = io.StringIO()
        old_stderr = sys.stderr
        try:
            sys.stderr = devnull
            with redirect_stderr(devnull):
                optimizer.optimize(
                    evaluation_budget=args.evaluation_budget,
                    starting_point=sequence,
                    rng_seed=rng_seed,
                )
        finally:
            sys.stderr = old_stderr


if __name__ == "__main__":
    main()

