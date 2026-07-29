"""Normalized CSV serialization for reusable local-enumeration catalogs."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pep_compass.local_enumeration.catalog import (
    CandidateOccurrence,
    CandidateOccurrenceSpool,
    LocalEnumerationTrace,
    normalize_tracking_level,
)


class LocalEnumerationCSVWriter:
    """Write normalized local-enumeration tables with bounded working memory."""

    def __init__(
        self,
        output_directory: str | Path,
        run_id: str,
        level: str,
        encoder_decoder: Any | None = None,
        store_candidate_latents: bool = True,
    ) -> None:
        """Initialize a run-local local-enumeration writer.

        :param output_directory: Directory receiving normalized CSV tables.
        :param run_id: Stable identifier included in every table.
        :param level: Tracking mode: ``quick``, ``normal``, or ``long``.
        :param encoder_decoder: Optional encoder for unique candidate means.
        :param store_candidate_latents: Whether candidate means are encoded.
        :raises ValueError: If ``level`` is unsupported.
        """
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.level = normalize_tracking_level(level)
        self.encoder_decoder = encoder_decoder
        self.store_candidate_latents = store_candidate_latents
        self._candidate_ids: dict[str, str] = {}
        self._written_candidates: set[str] = set()
        self._accepted_origins = CandidateOccurrenceSpool()
        self._write_headers()

    # Public API
    def candidate_id(self, sequence: str) -> str:
        """Return one stable run-local identifier for a peptide sequence.

        :param sequence: Peptide sequence to identify.
        :return: Stable identifier reused across local-enumeration tables.
        """
        if sequence not in self._candidate_ids:
            self._candidate_ids[sequence] = (
                f"candidate_{len(self._candidate_ids):06d}"
            )
        return self._candidate_ids[sequence]

    def write_iteration(
        self,
        iteration_id: int,
        trace: LocalEnumerationTrace,
        evaluated_sequences: set[str],
    ) -> None:
        """Write one catalog according to the configured tracking level.

        :param iteration_id: Optimizer iteration that produced the catalog.
        :param trace: Completed local-enumeration catalog.
        :param evaluated_sequences: Sequences evaluated after this iteration.
        """
        if self.level == "short":
            trace.all_candidates.close()
            return
        self._write_rows(
            "walker_steps.csv",
            self.step_fields(),
            (
                {"run_id": self.run_id, "iteration_id": iteration_id, **asdict(step)}
                for step in trace.steps
            ),
        )
        for occurrence in trace.all_candidates:
            if occurrence.passed_constraint_filter:
                if occurrence.source_iteration_id is None:
                    occurrence.source_iteration_id = iteration_id
                self._accepted_origins.append(occurrence)
        self._write_evaluated_origins(iteration_id, evaluated_sequences)
        if self.level == "all":
            self._write_unique_candidates(trace.post_cap_sequences)
            self._write_occurrences(iteration_id, trace.all_candidates)
        trace.all_candidates.close()

    def close(self) -> None:
        """Release temporary storage used for cross-iteration origins."""
        self._accepted_origins.close()

    # Schemas
    @staticmethod
    def step_fields() -> list[str]:
        """Return the stable walker-step schema."""
        return [
            "run_id",
            "iteration_id",
            "trajectory_id",
            "step_id",
            "node_id",
            "parent_id",
            "parent_sequence",
            "next_sequence",
            "theoretical_product_count",
            "post_cap_event_count",
            "post_cap_unique_count",
            "post_method_filter_event_count",
            "post_method_filter_unique_count",
            "post_constraint_event_count",
            "post_constraint_unique_count",
            "mutang_position_count",
            "mutang_residue_option_count",
            "latent_dimension",
            "effective_dimension",
            "current_latent_position",
            "next_latent_position",
            "singular_values",
            "adjusted_time_step",
        ]

    @staticmethod
    def origin_fields() -> list[str]:
        """Return the stable evaluated-origin schema."""
        return [
            "run_id",
            "evaluation_iteration_id",
            "source_iteration_id",
            "candidate_id",
            "trajectory_id",
            "step_id",
            "node_id",
            "parent_id",
        ]

    @staticmethod
    def unique_candidate_fields() -> list[str]:
        """Return the stable unique-candidate schema."""
        return [
            "run_id",
            "candidate_id",
            "sequence",
            "sequence_length",
            "encoded_latent_mean",
        ]

    @staticmethod
    def occurrence_fields() -> list[str]:
        """Return the stable candidate-occurrence schema."""
        return [
            "run_id",
            "iteration_id",
            "trajectory_id",
            "step_id",
            "candidate_id",
            "node_id",
            "parent_id",
            "passed_method_filter",
            "passed_constraint_filter",
            "method_score",
            "method_probability",
            "method_cumulative_probability",
            "method_rank",
            "levenshtein_to_center",
            "levenshtein_to_parent",
        ]

    # Serialization
    def _write_headers(self) -> None:
        if self.level == "short":
            return
        self._write_rows("walker_steps.csv", self.step_fields(), [])
        self._write_rows("evaluated_origins.csv", self.origin_fields(), [])
        if self.level == "all":
            self._write_rows(
                "unique_candidates.csv", self.unique_candidate_fields(), []
            )
            self._write_rows(
                "candidate_occurrences.csv", self.occurrence_fields(), []
            )

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

    def _encode(self, sequences: list[str]) -> dict[str, str]:
        if (
            not sequences
            or not self.store_candidate_latents
            or self.encoder_decoder is None
        ):
            return {}
        import torch

        with torch.no_grad():
            latent_points = self.encoder_decoder.encode_peptides(sequences)
        return {
            sequence: json.dumps(
                point.detach().cpu().tolist(), separators=(",", ":")
            )
            for sequence, point in zip(sequences, latent_points)
        }

    def _write_evaluated_origins(
        self, iteration_id: int, evaluated_sequences: set[str]
    ) -> None:
        self._write_rows(
            "evaluated_origins.csv",
            self.origin_fields(),
            (
                {
                    "run_id": self.run_id,
                    "evaluation_iteration_id": iteration_id,
                    "source_iteration_id": (
                        occurrence.source_iteration_id
                        if occurrence.source_iteration_id is not None
                        else iteration_id
                    ),
                    "candidate_id": self.candidate_id(occurrence.sequence),
                    "trajectory_id": occurrence.trajectory_id,
                    "step_id": occurrence.step_id,
                    "node_id": occurrence.node_id,
                    "parent_id": occurrence.parent_id,
                }
                for occurrence in self._accepted_origins
                if occurrence.sequence in evaluated_sequences
            ),
        )

    def _write_unique_candidates(self, sequences: set[str]) -> None:
        new_sequences = sorted(sequences - self._written_candidates)
        for start in range(0, len(new_sequences), 256):
            batch = new_sequences[start : start + 256]
            latent_points = self._encode(batch)
            self._write_rows(
                "unique_candidates.csv",
                self.unique_candidate_fields(),
                (
                    {
                        "run_id": self.run_id,
                        "candidate_id": self.candidate_id(sequence),
                        "sequence": sequence,
                        "sequence_length": len(sequence),
                        "encoded_latent_mean": latent_points.get(sequence, ""),
                    }
                    for sequence in batch
                ),
            )
        self._written_candidates.update(new_sequences)

    def _write_occurrences(
        self,
        iteration_id: int,
        occurrences: Iterable[CandidateOccurrence],
    ) -> None:
        self._write_rows(
            "candidate_occurrences.csv",
            self.occurrence_fields(),
            (
                {
                    "run_id": self.run_id,
                    "iteration_id": iteration_id,
                    "trajectory_id": occurrence.trajectory_id,
                    "step_id": occurrence.step_id,
                    "candidate_id": self.candidate_id(occurrence.sequence),
                    "node_id": occurrence.node_id,
                    "parent_id": occurrence.parent_id,
                    "passed_method_filter": occurrence.passed_method_filter,
                    "passed_constraint_filter": (
                        occurrence.passed_constraint_filter
                    ),
                    "method_score": occurrence.method_score,
                    "method_probability": occurrence.method_probability,
                    "method_cumulative_probability": (
                        occurrence.method_cumulative_probability
                    ),
                    "method_rank": occurrence.method_rank,
                    "levenshtein_to_center": occurrence.levenshtein_to_center,
                    "levenshtein_to_parent": occurrence.levenshtein_to_parent,
                }
                for occurrence in occurrences
            ),
        )
