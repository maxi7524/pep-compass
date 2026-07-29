"""CSV tracking for LE-BO candidate provenance and iteration summaries."""

from __future__ import annotations

import csv
import json
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

TrackingLevel = Literal["short", "normal", "all"]


@dataclass(slots=True)
class CandidateProvenance:
    """Describe the first local-enumeration path that produced a candidate.

    :param sequence: Candidate peptide sequence.
    :param parent_sequence: Sequence mutated to create the candidate.
    :param trajectory_id: Zero-based SORBES trajectory index.
    :param step_id: One-based step within the trajectory.
    :param path: Decoded SORBES path up to the generating step.
    """

    sequence: str
    parent_sequence: str
    trajectory_id: int | None
    step_id: int | None
    source_iteration_id: int | None = None
    node_id: str = ""
    parent_id: str = ""
    passed_method_filter: bool = True
    passed_constraint_filter: bool = True
    method_score: float | None = None
    method_probability: float | None = None
    method_cumulative_probability: float | None = None
    method_rank: int | None = None


class CandidateEventSpool:
    """Spill candidate events to disk instead of retaining an unbounded list."""

    def __init__(self, maximum_memory_bytes: int = 1_048_576) -> None:
        self.count = 0
        self._events = tempfile.SpooledTemporaryFile(  # noqa: SIM115
            mode="w+", max_size=maximum_memory_bytes, encoding="utf-8"
        )

    def append(self, candidate: CandidateProvenance) -> None:
        """Append one compact JSON event to the spool."""
        self._events.write(json.dumps(asdict(candidate), separators=(",", ":")))
        self._events.write("\n")
        self.count += 1

    def __iter__(self) -> Iterator[CandidateProvenance]:
        self._events.seek(0)
        for line in self._events:
            yield CandidateProvenance(**json.loads(line))

    def close(self) -> None:
        """Close and remove any temporary backing file."""
        self._events.close()


@dataclass(slots=True)
class EnumerationStep:
    """Store compact counts and walker state for one enumeration step."""

    trajectory_id: int | None
    step_id: int
    node_id: str
    parent_id: str
    parent_sequence: str
    next_sequence: str
    proposed_count: int
    post_limit_count: int
    post_method_filter_count: int
    post_constraint_filter_count: int
    current_latent_position: str = ""
    next_latent_position: str = ""
    adjusted_time_step: float | None = None


@dataclass
class EnumerationTrace:
    """Summarize one local-enumeration call with level-dependent provenance."""

    generated_count: int = 0
    accepted_count: int = 0
    candidates: dict[str, CandidateProvenance] = field(default_factory=dict)
    steps: list[EnumerationStep] = field(default_factory=list)
    all_candidates: CandidateEventSpool = field(default_factory=CandidateEventSpool)


class LeboCSVTracker:
    """Persist compact LE-BO histories in separate analysis-friendly files.

    ``short`` stores only APEX evaluations. ``normal`` adds proposal counts for
    every walker step and provenance for evaluated candidates. ``all`` stores
    every candidate after the Cartesian-product limit, including candidates
    rejected by the method-specific and final constraint filters.
    """

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
        if level not in {"short", "normal", "all"}:
            raise ValueError("tracking.level must be short, normal, or all")
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.level = level
        self.run_id = run_id
        self.candidate_strategy = candidate_strategy
        self.objective_name = objective_name
        self.objective_direction = objective_direction
        self.objective_description = objective_description
        self.objective_parameters = objective_parameters
        self.encoder_decoder = encoder_decoder
        self.store_latents = store_latents
        self.store_walker_latents = store_walker_latents
        self._candidate_ids: dict[str, str] = {}
        self._write_headers()
        self._write_metadata()

    def _write_headers(self) -> None:
        if self.level != "short":
            self._write_rows(
                "iteration_statistics.csv",
                [
                    "run_id",
                    "iteration_id",
                    "center_sequence",
                    "generated_count",
                    "accepted_count",
                    "rejected_count",
                    "evaluated_count",
                    "candidate_pool_size",
                    "best_sequence",
                    "best_objective_value",
                ],
                [],
            )
        self._write_rows(
            "evaluations.csv",
            [
                "run_id",
                "iteration_id",
                "candidate_id",
                "sequence",
                "objective_name",
                "objective_value",
                "objective_direction",
            ],
            [],
        )
        if self.level != "short":
            self._write_rows(
                "enumeration_steps.csv",
                [
                    "run_id",
                    "iteration_id",
                    "trajectory_id",
                    "step_id",
                    "node_id",
                    "parent_id",
                    "parent_sequence",
                    "next_sequence",
                    "proposed_count",
                    "post_limit_count",
                    "post_method_filter_count",
                    "post_constraint_filter_count",
                    "current_latent_position",
                    "next_latent_position",
                    "adjusted_time_step",
                ],
                [],
            )
            self._write_rows(
                "candidates.csv",
                [
                    "run_id",
                    "iteration_id",
                    "candidate_id",
                    "sequence",
                    "parent_sequence",
                    "trajectory_id",
                    "step_id",
                    "source_iteration_id",
                    "node_id",
                    "parent_id",
                    "passed_method_filter",
                    "passed_constraint_filter",
                    "method_score",
                    "method_probability",
                    "method_cumulative_probability",
                    "method_rank",
                    "latent_point",
                    "evaluated",
                ],
                [],
            )

    def _write_metadata(self) -> None:
        metadata = {
            "tracking_level": self.level,
            "run_id": self.run_id,
            "candidate_strategy": self.candidate_strategy,
            "objective_name": self.objective_name,
            "objective_direction": self.objective_direction,
            "objective_description": self.objective_description,
            "objective_parameters": self.objective_parameters,
            "store_latents": self.store_latents,
            "store_walker_latents": self.store_walker_latents,
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

    def _encode(self, sequences: list[str]) -> dict[str, str]:
        if not sequences or not self.store_latents or self.encoder_decoder is None:
            return {}
        import torch

        encoded: dict[str, str] = {}
        with torch.no_grad():
            for start in range(0, len(sequences), 256):
                batch = sequences[start : start + 256]
                latent_points = self.encoder_decoder.encode_peptides(batch)
                encoded.update(
                    {
                        sequence: json.dumps(point.detach().cpu().tolist())
                        for sequence, point in zip(batch, latent_points)
                    }
                )
        return encoded

    def _candidate_id(self, sequence: str) -> str:
        """Return one stable run-local identifier for a peptide sequence."""
        if sequence not in self._candidate_ids:
            self._candidate_ids[sequence] = f"candidate_{len(self._candidate_ids):06d}"
        return self._candidate_ids[sequence]

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
        """Append one optimizer iteration and its configured candidate subset."""
        evaluated = {sequence for sequence, _ in evaluations}
        if self.level != "short":
            self._write_rows(
                "iteration_statistics.csv",
                [
                    "run_id",
                    "iteration_id",
                    "center_sequence",
                    "generated_count",
                    "accepted_count",
                    "rejected_count",
                    "evaluated_count",
                    "candidate_pool_size",
                    "best_sequence",
                    "best_objective_value",
                ],
                [
                    {
                        "run_id": self.run_id,
                        "iteration_id": iteration_id,
                        "center_sequence": center_sequence,
                        "generated_count": trace.generated_count,
                        "accepted_count": trace.accepted_count,
                        "rejected_count": max(
                            trace.generated_count - trace.accepted_count, 0
                        ),
                        "evaluated_count": len(evaluations),
                        "candidate_pool_size": candidate_pool_size,
                        "best_sequence": best_sequence,
                        "best_objective_value": best_objective_value,
                    }
                ],
            )
        self._write_rows(
            "evaluations.csv",
            [
                "run_id",
                "iteration_id",
                "candidate_id",
                "sequence",
                "objective_name",
                "objective_value",
                "objective_direction",
            ],
            (
                {
                    "run_id": self.run_id,
                    "iteration_id": iteration_id,
                    "candidate_id": self._candidate_id(sequence),
                    "sequence": sequence,
                    "objective_name": self.objective_name,
                    "objective_value": score,
                    "objective_direction": self.objective_direction,
                }
                for sequence, score in evaluations
            ),
        )
        if self.level == "short":
            trace.all_candidates.close()
            return
        self._write_rows(
            "enumeration_steps.csv",
            [
                "run_id",
                "iteration_id",
                "trajectory_id",
                "step_id",
                "node_id",
                "parent_id",
                "parent_sequence",
                "next_sequence",
                "proposed_count",
                "post_limit_count",
                "post_method_filter_count",
                "post_constraint_filter_count",
                "current_latent_position",
                "next_latent_position",
                "adjusted_time_step",
            ],
            [
                {"run_id": self.run_id, "iteration_id": iteration_id, **asdict(step)}
                for step in trace.steps
            ],
        )
        provenance = (
            trace.candidates if candidate_provenance is None else candidate_provenance
        )
        selected = (
            trace.all_candidates
            if self.level == "all"
            else list(
                {
                    sequence: provenance[sequence]
                    for sequence in evaluated
                    if sequence in provenance
                }.values()
            )
        )
        latent_points = (
            self._encode(list({item.sequence for item in selected}))
            if self.store_latents
            else {}
        )
        self._write_rows(
            "candidates.csv",
            [
                "run_id",
                "iteration_id",
                "candidate_id",
                "sequence",
                "parent_sequence",
                "trajectory_id",
                "step_id",
                "source_iteration_id",
                "node_id",
                "parent_id",
                "passed_method_filter",
                "passed_constraint_filter",
                "method_score",
                "method_probability",
                "method_cumulative_probability",
                "method_rank",
                "latent_point",
                "evaluated",
            ],
            (
                {
                    "run_id": self.run_id,
                    "iteration_id": iteration_id,
                    "candidate_id": self._candidate_id(item.sequence),
                    "sequence": item.sequence,
                    "parent_sequence": item.parent_sequence,
                    "trajectory_id": item.trajectory_id,
                    "step_id": item.step_id,
                    "source_iteration_id": (
                        item.source_iteration_id
                        if item.source_iteration_id is not None
                        else iteration_id
                    ),
                    "node_id": item.node_id,
                    "parent_id": item.parent_id,
                    "passed_method_filter": item.passed_method_filter,
                    "passed_constraint_filter": item.passed_constraint_filter,
                    "method_score": item.method_score,
                    "method_probability": item.method_probability,
                    "method_cumulative_probability": item.method_cumulative_probability,
                    "method_rank": item.method_rank,
                    "latent_point": latent_points.get(item.sequence, ""),
                    "evaluated": item.sequence in evaluated,
                }
                for item in selected
            ),
        )
        trace.all_candidates.close()
