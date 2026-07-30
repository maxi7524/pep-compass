"""Experiment wrapper for configuration-defined optimization trees."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from pep_compass.core.builder import PepCompassCore
from pep_compass.optimization.result import OptimizationResult
from pep_compass.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


@dataclass
class ComposableExperiment:
    """Run and persist one composable optimization experiment."""

    config: Mapping[str, Any]
    core: PepCompassCore

    def run(self) -> OptimizationResult:
        """Execute configured steps and persist their final candidate batch."""
        experiment = self.config.get("experiment", {})
        input_config = experiment.get("input", {})
        sequences = input_config.get("sequences")
        if not isinstance(sequences, list) or not all(
            isinstance(sequence, str) for sequence in sequences
        ):
            raise ValueError("experiment.input.sequences must be a list of strings.")
        result = self.core.build_runner(self.config).run(
            sequences,
            seed=experiment.get("seed"),
        )
        output = experiment.get("output", {})
        directory = output.get("directory")
        if directory is not None:
            self._persist(result, Path(directory))
        return result

    def _persist(self, result: OptimizationResult, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "candidates.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(("candidate_index", "sequence"))
            writer.writerows(enumerate(result.candidates.sequences))
        torch.save(
            result.candidates.latent_origins.detach().cpu(),
            directory / "latent_origins.pt",
        )
        summary = {
            "status": "completed",
            "final_candidate_count": len(result.candidates),
            "objective_name": result.objective_name,
            "objective_direction": result.objective_direction,
            "best_sequence": (
                result.best_candidate.sequence
                if result.best_candidate is not None
                else None
            ),
            "best_score": result.best_score,
        }
        with (directory / "result.json").open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2)
        logger.info("Persisted composable experiment result in %s.", directory)
