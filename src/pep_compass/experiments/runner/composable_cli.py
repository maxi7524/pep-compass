"""Command-line runner for composable optimization configurations."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from pep_compass.core import PepCompassCore, load_configuration, validate_configuration
from pep_compass.experiments.composable import ComposableExperiment
from pep_compass.experiments.plan import materialize_execution_plan
from pep_compass.experiments.backends import execute_subprocess_plan, write_slurm_array_script
from pep_compass.optimization.tracking import CSVStepTracker


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a composable PepCompass optimization experiment."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--run-index", type=int, action="append")
    parser.add_argument("--backend", choices=("local", "subprocess", "slurm"))
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--slurm-script", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--working-directory", type=Path)
    return parser.parse_args()


def main() -> None:
    """Load, construct, and execute one composable experiment."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()
    working_directory = (
        Path.cwd()
        if args.working_directory is None
        else args.working_directory.resolve()
    )
    config_path = args.config.resolve()
    config = load_configuration(config_path)
    validate_configuration(config)
    experiment = config.get("experiment", {})
    if args.dry_run:
        plan = materialize_execution_plan(
            config, working_directory=working_directory
        )
        print(
            json.dumps(
                {
                    "total_runs": len(plan),
                    "runs": [
                        {
                            "index": entry.index,
                            "variant_id": entry.variant.variant_id,
                            "run_id": entry.task.run_id,
                            "seed": entry.seed,
                            "sequence": entry.task.sequence,
                        }
                        for entry in plan
                    ],
                },
                indent=2,
            )
        )
        return
    plan = materialize_execution_plan(
        config, working_directory=working_directory
    )
    execution = experiment.get("execution", {})
    backend = args.backend or execution.get("backend", "local")
    max_workers = args.max_workers or execution.get("max_workers", 1)
    if args.run_index is not None:
        requested = set(args.run_index)
        plan = tuple(entry for entry in plan if entry.index in requested)
        if len(plan) != len(requested):
            raise ValueError("At least one requested run index is outside the plan.")
    if backend == "subprocess" and not args.worker:
        execute_subprocess_plan(
            plan,
            config_path,
            working_directory=working_directory,
            max_workers=max_workers,
            resume=args.resume,
            continue_on_error=args.continue_on_error,
        )
        return
    if backend == "slurm" and not args.worker:
        if args.slurm_script is None:
            raise ValueError("--slurm-script is required for the Slurm backend.")
        path = write_slurm_array_script(
            plan,
            config_path,
            args.slurm_script,
            working_directory=working_directory,
            settings=execution.get("slurm", {}),
            resume=args.resume,
            continue_on_error=args.continue_on_error,
        )
        print(path)
        return
    tracking = experiment.get("tracking")

    def tracker_factory(run_directory: Path, task, variant):
        if tracking is None:
            return None
        return CSVStepTracker(
            run_directory / "tracking",
            level=tracking.get("level", "normal"),
            max_depth=tracking.get("max_depth"),
            store_latents=tracking.get("store_latents", False),
            store_fields=tracking.get("store_fields", False),
            run_id=task.run_id,
            variant_id=variant.variant_id,
        )
    core = PepCompassCore.from_config(config)
    ComposableExperiment(
        config,
        core,
        working_directory=working_directory,
        tracker_factory=tracker_factory,
        resume=args.resume,
        on_error="continue" if args.continue_on_error else "stop",
        run_indices=set(args.run_index) if args.run_index is not None else None,
        persist_manifest=not args.worker,
    ).run()


if __name__ == "__main__":
    main()
