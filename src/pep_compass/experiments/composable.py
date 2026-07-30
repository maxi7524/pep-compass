"""Experiment orchestration for independently executable optimization tasks."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from pep_compass.core.builder import PepCompassCore
from pep_compass.experiments.input import ExperimentTask, materialize_input_tasks
from pep_compass.experiments.variants import ExperimentVariant, materialize_variants
from pep_compass.optimization.result import OptimizationResult
from pep_compass.optimization.tracking import StepTracker
from pep_compass.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)

TrackerFactory = Callable[[Path], StepTracker | None]


@dataclass(frozen=True)
class ExperimentRun:
    """Result and identity of one independent optimization run."""

    task: ExperimentTask
    variant: ExperimentVariant
    seed: int | None
    result: OptimizationResult
    output_directory: Path | None


@dataclass(frozen=True)
class ExperimentResult:
    """Collection of independently executed experiment runs."""

    runs: tuple[ExperimentRun, ...]


@dataclass
class ComposableExperiment:
    """Materialize, execute, and persist a composable experiment."""

    config: Mapping[str, Any]
    core: PepCompassCore
    config_directory: Path | None = None
    tracker_factory: TrackerFactory | None = None

    def run(self) -> ExperimentResult:
        """Execute every input repetition as an independent optimization run.

        The encoder-decoder owned by ``core`` is reused, while runner state,
        random state, budgets, tracking, and outputs are isolated per task.

        :return: Ordered independent run results.
        :rtype: ExperimentResult
        """
        experiment = self.config.get("experiment", {})
        tasks = materialize_input_tasks(
            experiment.get("input", {}),
            base_directory=self.config_directory,
        )
        output_root = self._resolve_output_directory(experiment.get("output", {}))
        base_seed = experiment.get("seed")
        variants = materialize_variants(self.config)
        runs: list[ExperimentRun] = []
        for variant, task in (
            (variant, task) for variant in variants for task in tasks
        ):
            multiple_variants = len(variants) > 1
            variant_directory = (
                output_root / "variants" / variant.variant_id
                if output_root is not None and multiple_variants
                else output_root
            )
            run_directory = (
                variant_directory / "runs" / task.run_id
                if variant_directory is not None
                else None
            )
            tracker = (
                self.tracker_factory(run_directory)
                if self.tracker_factory is not None and run_directory is not None
                else None
            )
            run_index = variant.index * len(tasks) + task.index
            seed = base_seed + run_index if base_seed is not None else None
            logger.info(
                "Executing variant %s task %s for source %s repetition %s.",
                variant.variant_id,
                task.run_id,
                task.source_index,
                task.repetition,
            )
            result = self.core.build_runner(variant.config, tracker=tracker).run(
                [task.sequence], seed=seed
            )
            if run_directory is not None:
                self._persist_run(result, task, variant, seed, run_directory)
            runs.append(ExperimentRun(task, variant, seed, result, run_directory))
        experiment_result = ExperimentResult(tuple(runs))
        if output_root is not None:
            self._persist_manifest(experiment_result, output_root)
        return experiment_result

    def _resolve_output_directory(
        self, configuration: Mapping[str, Any]
    ) -> Path | None:
        directory = configuration.get("directory")
        if directory is None:
            return None
        resolved = Path(directory)
        if not resolved.is_absolute() and self.config_directory is not None:
            resolved = self.config_directory / resolved
        return resolved

    @staticmethod
    def _persist_run(
        result: OptimizationResult,
        task: ExperimentTask,
        variant: ExperimentVariant,
        seed: int | None,
        directory: Path,
    ) -> None:
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
            "run_id": task.run_id,
            "variant_id": variant.variant_id,
            "variant_values": dict(variant.values),
            "source_index": task.source_index,
            "repetition": task.repetition,
            "seed": seed,
            "input_sequence": task.sequence,
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

    @staticmethod
    def _persist_manifest(result: ExperimentResult, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "run_manifest.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(
                (
                    "run_id",
                    "variant_id",
                    "variant_values",
                    "source_index",
                    "repetition",
                    "seed",
                    "input_sequence",
                    "final_candidate_count",
                    "best_sequence",
                    "best_score",
                    "objective_name",
                    "objective_direction",
                )
            )
            for run in result.runs:
                writer.writerow(
                    (
                        run.task.run_id,
                        run.variant.variant_id,
                        json.dumps(dict(run.variant.values), separators=(",", ":")),
                        run.task.source_index,
                        run.task.repetition,
                        run.seed,
                        run.task.sequence,
                        len(run.result.candidates),
                        run.result.best_candidate.sequence
                        if run.result.best_candidate is not None
                        else None,
                        run.result.best_score,
                        run.result.objective_name,
                        run.result.objective_direction,
                    )
                )
        logger.info("Persisted %s experiment runs in %s.", len(result.runs), directory)
