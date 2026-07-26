"""Run configured peptide-optimization experiments locally or through Slurm."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

from configuration import load_config, load_sequences, prepare_tasks
from execution import run_local_tasks, run_srun_tasks, run_task_file

logger = logging.getLogger(__name__)

def _parse_args() -> argparse.Namespace:
    """Parse public runner options and the internal task-file entry point."""
    parser = argparse.ArgumentParser(
        description="Run configured peptide optimization grids locally or with srun."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--task-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--device", help="Override the configured Torch device.")
    parser.add_argument("--output", type=Path, help="Override the output root.")
    parser.add_argument("--budget", type=int, help="Override evaluations per run.")
    parser.add_argument("--seed", type=int, help="Override the base seed.")
    parser.add_argument("--execution", choices=["local", "srun"])
    parser.add_argument("--max-parallel-runs", type=int)
    parser.add_argument(
        "--devices",
        nargs="+",
        help="Assign parallel tasks round-robin to these Torch devices.",
    )
    parser.add_argument(
        "--srun-argument",
        action="append",
        help="Replace configured srun arguments; repeat once per argument.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and materialize all grid tasks without loading models.",
    )
    return parser.parse_args()


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    """Apply CLI overrides and remove conflicting grid entries.

    :param config: Loaded configuration modified in place.
    :param args: Parsed runner arguments.
    """
    overrides = {
        "device": args.device,
        "output_path": str(args.output) if args.output is not None else None,
        "evaluation_budget": args.budget,
        "seed": args.seed,
    }
    for path, value in overrides.items():
        if value is not None:
            config[path] = value
            config.get("grid", {}).pop(path, None)
    if args.execution is not None:
        config["execution"]["backend"] = args.execution
    if args.max_parallel_runs is not None:
        config["execution"]["max_parallel_runs"] = args.max_parallel_runs
    if args.devices is not None:
        config["execution"]["devices"] = args.devices
    if args.srun_argument is not None:
        config["execution"]["srun"]["arguments"] = args.srun_argument


def main() -> None:
    """Materialize a configured experiment and execute its tasks."""
    args = _parse_args()
    if args.task_file is not None:
        run_task_file(args.task_file)
        return
    config = load_config(args.config)
    _apply_overrides(config, args)
    if config.get("seed") is None:
        config["seed"] = int(time.time())
    output_root = Path(config["output_path"])
    output_root.mkdir(parents=True, exist_ok=True)
    tasks, task_paths = prepare_tasks(
        config, load_sequences(Path(config["input_csv"])), output_root
    )
    logger.info("Prepared %s runs in %s", len(tasks), output_root)
    if args.dry_run:
        return
    if config["execution"]["backend"] == "srun":
        run_srun_tasks(task_paths, config["execution"])
    else:
        run_local_tasks(tasks, task_paths, config["execution"])


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()


