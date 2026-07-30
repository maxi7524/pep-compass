"""Command-line runner for composable optimization configurations."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from pep_compass.core import PepCompassCore, load_configuration, validate_configuration
from pep_compass.experiments.composable import ComposableExperiment
from pep_compass.experiments.input import materialize_input_tasks
from pep_compass.experiments.variants import materialize_variants
from pep_compass.optimization.tracking import CSVStepTracker


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a composable PepCompass optimization experiment."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Load, construct, and execute one composable experiment."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()
    config_path = args.config.resolve()
    config = load_configuration(config_path)
    validate_configuration(config)
    experiment = config.get("experiment", {})
    if args.dry_run:
        tasks = materialize_input_tasks(
            experiment.get("input", {}), base_directory=config_path.parent
        )
        variants = materialize_variants(config)
        print(
            json.dumps(
                {
                    "variants": len(variants),
                    "tasks_per_variant": len(tasks),
                    "total_runs": len(variants) * len(tasks),
                },
                indent=2,
            )
        )
        return
    tracking = experiment.get("tracking")

    def tracker_factory(run_directory: Path):
        if tracking is None:
            return None
        return CSVStepTracker(
            run_directory / "tracking",
            level=tracking.get("level", "normal"),
            max_depth=tracking.get("max_depth"),
            store_latents=tracking.get("store_latents", False),
            store_fields=tracking.get("store_fields", False),
        )
    core = PepCompassCore.from_config(config)
    ComposableExperiment(
        config,
        core,
        config_directory=config_path.parent,
        tracker_factory=tracker_factory,
        resume=args.resume,
        on_error="continue" if args.continue_on_error else "stop",
    ).run()


if __name__ == "__main__":
    main()
