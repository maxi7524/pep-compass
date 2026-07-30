"""Experiment orchestration for independently executable optimization tasks."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from pep_compass.core.builder import PepCompassCore
from pep_compass.experiments.input import ExperimentTask, materialize_input_tasks
from pep_compass.experiments.variants import ExperimentVariant, materialize_variants
from pep_compass.optimization.result import OptimizationResult
from pep_compass.optimization.tracking import StepTracker
from pep_compass.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)

TrackerFactory = Callable[[Path], StepTracker | None]
RunStatus = Literal["completed", "failed", "skipped"]


@dataclass(frozen=True)
class ExperimentRun:
    """Result and identity of one independent optimization run."""

    task: ExperimentTask
    variant: ExperimentVariant
    seed: int | None
    status: RunStatus
    result: OptimizationResult | None
    output_directory: Path | None
    error: str | None = None


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
    resume: bool = False
    on_error: Literal["stop", "continue"] = "stop"

    def run(self) -> ExperimentResult:
        """Execute every input repetition as an independent optimization run.

        The encoder-decoder owned by ``core`` is reused, while runner state,
        random state, budgets, tracking, and outputs are isolated per task.

        :return: Ordered independent run results.
        :rtype: ExperimentResult
        """
        if self.on_error not in {"stop", "continue"}:
            raise ValueError("Experiment on_error must be stop or continue.")
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
            run_index = variant.index * len(tasks) + task.index
            seed = base_seed + run_index if base_seed is not None else None
            if self.resume and run_directory is not None and self._is_completed(run_directory):
                logger.info("Skipping completed run %s/%s.", variant.variant_id, task.run_id)
                runs.append(
                    ExperimentRun(
                        task, variant, seed, "skipped", None, run_directory
                    )
                )
                continue
            tracker = (
                self.tracker_factory(run_directory)
                if self.tracker_factory is not None and run_directory is not None
                else None
            )
            logger.info(
                "Executing variant %s task %s for source %s repetition %s.",
                variant.variant_id,
                task.run_id,
                task.source_index,
                task.repetition,
            )
            if run_directory is not None:
                self._persist_status("running", task, variant, seed, run_directory)
            try:
                result = self.core.build_runner(variant.config, tracker=tracker).run(
                    [task.sequence], seed=seed
                )
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                logger.error("Experiment run %s failed: %s", task.run_id, message)
                if run_directory is not None:
                    self._persist_status(
                        "failed", task, variant, seed, run_directory, error=message
                    )
                runs.append(
                    ExperimentRun(
                        task, variant, seed, "failed", None, run_directory, message
                    )
                )
                if self.on_error == "stop":
                    partial = ExperimentResult(tuple(runs))
                    if output_root is not None:
                        self._persist_manifest(partial, output_root)
                    raise
                continue
            if run_directory is not None:
                self._persist_run(result, task, variant, seed, run_directory)
            runs.append(
                ExperimentRun(
                    task, variant, seed, "completed", result, run_directory
                )
            )
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
        temporary = directory / "result.json.tmp"
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2)
        temporary.replace(directory / "result.json")

    @staticmethod
    def _persist_status(
        status: str,
        task: ExperimentTask,
        variant: ExperimentVariant,
        seed: int | None,
        directory: Path,
        *,
        error: str | None = None,
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        summary = {
            "status": status,
            "run_id": task.run_id,
            "variant_id": variant.variant_id,
            "variant_values": dict(variant.values),
            "source_index": task.source_index,
            "repetition": task.repetition,
            "seed": seed,
            "input_sequence": task.sequence,
            "error": error,
        }
        temporary = directory / "result.json.tmp"
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2)
        temporary.replace(directory / "result.json")

    @staticmethod
    def _is_completed(directory: Path) -> bool:
        result_path = directory / "result.json"
        if not result_path.exists():
            return False
        try:
            with result_path.open(encoding="utf-8") as stream:
                return json.load(stream).get("status") == "completed"
        except (OSError, json.JSONDecodeError):
            return False

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
                    "status",
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
                    "error",
                )
            )
            for run in result.runs:
                writer.writerow(
                    (
                        run.task.run_id,
                        run.status,
                        run.variant.variant_id,
                        json.dumps(dict(run.variant.values), separators=(",", ":")),
                        run.task.source_index,
                        run.task.repetition,
                        run.seed,
                        run.task.sequence,
                        len(run.result.candidates) if run.result is not None else None,
                        run.result.best_candidate.sequence
                        if run.result is not None and run.result.best_candidate is not None
                        else None,
                        run.result.best_score if run.result is not None else None,
                        run.result.objective_name if run.result is not None else None,
                        run.result.objective_direction if run.result is not None else None,
                        run.error,
                    )
                )
        logger.info("Persisted %s experiment runs in %s.", len(result.runs), directory)
