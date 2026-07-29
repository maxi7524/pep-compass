"""LEBO adapter for objective and local-enumeration experiment tracking."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from pep_compass.local_enumeration.catalog import (
    CandidateEventSpool,
    CandidateProvenance,
    EnumerationStep,
    EnumerationTrace,
    normalize_tracking_level,
)
from pep_compass.local_enumeration.catalog_writer import LocalEnumerationCSVWriter

TrackingLevel = Literal["quick", "normal", "long", "short", "all"]


class LeboCSVTracker:
    """Combine LEBO iteration results with a reusable enumeration writer."""

    def __init__(
        self,
        output_directory: str | Path,
        run_id: str,
        level: TrackingLevel,
        candidate_strategy: str,
        objective_name: str,
        objective_direction: str,
        objective_description: str,
        objective_parameters: dict[str, Any],
        encoder_decoder: Any | None = None,
        store_latents: bool = True,
        store_walker_latents: bool = False,
    ) -> None:
        """Initialize one LEBO tracking adapter.

        :param output_directory: Directory receiving tracking tables.
        :param run_id: Stable run identifier.
        :param level: Tracking mode: ``quick``, ``normal``, or ``long``.
        :param candidate_strategy: LEBO local candidate strategy.
        :param objective_name: Human-readable objective name.
        :param objective_direction: ``maximize`` or ``minimize``.
        :param objective_description: Objective semantics stored in metadata.
        :param objective_parameters: Resolved black-box configuration.
        :param encoder_decoder: Encoder used for unique candidate means.
        :param store_latents: Whether unique candidate means are stored.
        :param store_walker_latents: Whether walker geometry is stored.
        :raises ValueError: If ``level`` is unsupported.
        """
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.level = normalize_tracking_level(level)
        self.run_id = run_id
        self.candidate_strategy = candidate_strategy
        self.objective_name = objective_name
        self.objective_direction = objective_direction
        self.objective_description = objective_description
        self.objective_parameters = objective_parameters
        self.store_latents = store_latents
        self.store_walker_latents = store_walker_latents
        self.catalog_writer = LocalEnumerationCSVWriter(
            output_directory=self.output_directory,
            run_id=run_id,
            level=self.level,
            encoder_decoder=encoder_decoder,
            store_candidate_latents=store_latents,
        )
        self._write_headers()
        self._write_metadata()

    # Schemas and output primitives
    @staticmethod
    def _evaluation_fields() -> list[str]:
        return [
            "run_id",
            "iteration_id",
            "candidate_id",
            "sequence",
            "objective_name",
            "objective_value",
            "objective_direction",
        ]

    @staticmethod
    def _iteration_fields() -> list[str]:
        return [
            "run_id",
            "iteration_id",
            "center_sequence",
            "trajectory_count",
            "walker_step_count",
            "post_cap_event_count",
            "post_cap_unique_count",
            "post_method_filter_event_count",
            "post_method_filter_unique_count",
            "post_constraint_event_count",
            "post_constraint_unique_count",
            "optimizer_pool_unique_count",
            "evaluated_count",
            "best_sequence",
            "best_objective_value",
        ]

    def _write_headers(self) -> None:
        self._write_rows("evaluations.csv", self._evaluation_fields(), [])
        if self.level != "short":
            self._write_rows(
                "optimizer_iterations.csv", self._iteration_fields(), []
            )

    def _write_metadata(self) -> None:
        metadata = {
            "schema_version": 2,
            "tracking_level": self.level,
            "run_id": self.run_id,
            "candidate_strategy": self.candidate_strategy,
            "objective_name": self.objective_name,
            "objective_direction": self.objective_direction,
            "objective_description": self.objective_description,
            "objective_parameters": self.objective_parameters,
            "store_latents": self.store_latents,
            "store_walker_latents": self.store_walker_latents,
            "candidate_latent_semantics": "encoder_posterior_mean",
            "walker_latent_semantics": "actual_sorbes_position",
        }
        with (self.output_directory / "tracking_metadata.json").open(
            "w", encoding="utf-8"
        ) as output:
            json.dump(metadata, output, indent=2)

    def _write_rows(
        self,
        filename: str,
        fieldnames: list[str],
        rows: Iterable[dict[str, Any]],
    ) -> None:
        path = self.output_directory / filename
        write_header = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    # LEBO adapter
    def record_iteration(
        self,
        iteration_id: int,
        center_sequence: str,
        trace: EnumerationTrace,
        evaluations: list[tuple[str, float]],
        candidate_pool_size: int,
        best_sequence: str,
        best_objective_value: float,
        candidate_provenance: dict[str, CandidateProvenance] | None = None,
    ) -> None:
        """Append one optimizer iteration and its enumeration catalog.

        :param iteration_id: Zero-based initialization or optimizer iteration.
        :param center_sequence: Sequence around which local enumeration ran.
        :param trace: Catalog produced by the local enumerator.
        :param evaluations: Sequence and objective-value pairs evaluated now.
        :param candidate_pool_size: Cumulative unique optimizer pool size.
        :param best_sequence: Best sequence observed after the iteration.
        :param best_objective_value: Objective value of ``best_sequence``.
        :param candidate_provenance: Deprecated compatibility argument.
        """
        del candidate_provenance
        self._write_evaluations(iteration_id, evaluations)
        if self.level != "short":
            self._write_iteration_summary(
                iteration_id,
                center_sequence,
                trace,
                len(evaluations),
                candidate_pool_size,
                best_sequence,
                best_objective_value,
            )
        self.catalog_writer.write_iteration(
            iteration_id,
            trace,
            {sequence for sequence, _ in evaluations},
        )

    def _write_evaluations(
        self, iteration_id: int, evaluations: list[tuple[str, float]]
    ) -> None:
        self._write_rows(
            "evaluations.csv",
            self._evaluation_fields(),
            (
                {
                    "run_id": self.run_id,
                    "iteration_id": iteration_id,
                    "candidate_id": self.catalog_writer.candidate_id(sequence),
                    "sequence": sequence,
                    "objective_name": self.objective_name,
                    "objective_value": score,
                    "objective_direction": self.objective_direction,
                }
                for sequence, score in evaluations
            ),
        )

    def _write_iteration_summary(
        self,
        iteration_id: int,
        center_sequence: str,
        trace: EnumerationTrace,
        evaluated_count: int,
        candidate_pool_size: int,
        best_sequence: str,
        best_objective_value: float,
    ) -> None:
        self._write_rows(
            "optimizer_iterations.csv",
            self._iteration_fields(),
            [
                {
                    "run_id": self.run_id,
                    "iteration_id": iteration_id,
                    "center_sequence": center_sequence,
                    "trajectory_count": len(
                        {step.trajectory_id for step in trace.steps}
                    ),
                    "walker_step_count": len(trace.steps),
                    "post_cap_event_count": trace.post_cap_event_count,
                    "post_cap_unique_count": len(trace.post_cap_sequences),
                    "post_method_filter_event_count": (
                        trace.post_method_filter_event_count
                    ),
                    "post_method_filter_unique_count": len(
                        trace.post_method_filter_sequences
                    ),
                    "post_constraint_event_count": trace.post_constraint_event_count,
                    "post_constraint_unique_count": len(
                        trace.post_constraint_sequences
                    ),
                    "optimizer_pool_unique_count": candidate_pool_size,
                    "evaluated_count": evaluated_count,
                    "best_sequence": best_sequence,
                    "best_objective_value": best_objective_value,
                }
            ],
        )


__all__ = [
    "CandidateEventSpool",
    "CandidateProvenance",
    "EnumerationStep",
    "EnumerationTrace",
    "LeboCSVTracker",
]
