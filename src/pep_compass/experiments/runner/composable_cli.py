"""Command-line runner for composable optimization configurations."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pep_compass.core import PepCompassCore, load_configuration
from pep_compass.experiments.composable import ComposableExperiment
from pep_compass.optimization.tracking import CSVStepTracker


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a composable PepCompass optimization experiment."
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Load, construct, and execute one composable experiment."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    config_path = _parse_args().config.resolve()
    config = load_configuration(config_path)
    experiment = config.get("experiment", {})
    output = experiment.get("output", {})
    output_directory = Path(output.get("directory", "results/composable"))
    if not output_directory.is_absolute():
        output_directory = config_path.parent / output_directory
    tracking = experiment.get("tracking")
    tracker = None
    if tracking is not None:
        tracker = CSVStepTracker(
            output_directory / "tracking",
            level=tracking.get("level", "normal"),
            max_depth=tracking.get("max_depth"),
            store_latents=tracking.get("store_latents", False),
            store_fields=tracking.get("store_fields", False),
        )
    core = PepCompassCore.from_config(config, tracker=tracker)
    ComposableExperiment(config, core, config_directory=config_path.parent).run()


if __name__ == "__main__":
    main()
