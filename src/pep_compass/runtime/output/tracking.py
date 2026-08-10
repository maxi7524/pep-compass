"""CSV-backed consumer of lifecycle events emitted by ``Step.__call__``.

``RuntimeRunner._build_tracker`` selects this implementation when a run has an
output directory. The engine sees only the ``StepTracker`` contract and has no
knowledge of CSV files or retention settings.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import TYPE_CHECKING, Any

import torch

from pep_compass.optimization.tracking import ExecutionScope, StepTracker, TrackingLevel

if TYPE_CHECKING:
    from pep_compass.data.optimization import CandidateBatch
    from pep_compass.optimization.engine.execution.context import OptimizationContext
    from pep_compass.optimization.engine.execution.step import Step


class CSVStepTracker(StepTracker):
    """Stream depth-aware optimization records to normalized CSV files.

    ``short`` writes oracle summaries and candidates. ``normal`` writes scalar
    summaries for every enabled step and candidates for oracles. ``all`` also
    writes candidates for every enabled step. ``store_latents`` and
    ``store_fields`` control columns within candidate rows; ``field_names``
    restricts serialized fields further.

    Rows are serialized and flushed immediately. The tracker does not retain
    completed ``CandidateBatch`` objects, so Python and PyTorch may release
    their storage once no operation references them. Closing is guaranteed by
    ``PepCompassPipeline.run`` in a ``finally`` block.
    """

    def __init__(
        self,
        output_directory: str | Path,
        *,
        level: TrackingLevel = "normal",
        max_depth: int | None = None,
        store_latents: bool = False,
        store_fields: bool = False,
        field_names: list[str] | tuple[str, ...] | None = None,
        run_id: str | None = None,
        variant_id: str | None = None,
    ) -> None:
        if level not in {"short", "normal", "all"}:
            raise ValueError("Tracking level must be short, normal, or all.")
        self.level = level
        self.max_depth = max_depth
        self.store_latents = store_latents
        self.store_fields = store_fields
        self.field_names = frozenset(field_names) if field_names is not None else None
        self.run_id = run_id
        self.variant_id = variant_id
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._next_execution_id = 0
        self._step_stream = (self.output_directory / "steps.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self._candidate_stream = (self.output_directory / "candidates.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self._trajectory_stream = (self.output_directory / "trajectory_points.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self._local_enumeration_stream = (
            self.output_directory / "local_enumerations.csv"
        ).open("w", newline="", encoding="utf-8")
        self._trajectory_latents: list[torch.Tensor] = []
        self._local_input_latents: list[torch.Tensor] = []
        self._step_writer = csv.DictWriter(
            self._step_stream,
            fieldnames=(
                "execution_id",
                "run_id",
                "variant_id",
                "step_name",
                "path",
                "depth",
                "loop_indices",
                "branch_names",
                "branch_indices",
                "input_size",
                "output_size",
                "status",
                "duration_seconds",
                "oracle_calls_before",
                "oracle_calls_after",
                "generated_candidates_before",
                "generated_candidates_after",
                "error",
            ),
        )
        self._candidate_writer = csv.DictWriter(
            self._candidate_stream,
            fieldnames=(
                "execution_id",
                "candidate_index",
                "sequence",
                "latent_origin",
                "fields",
            ),
        )
        self._trajectory_writer = csv.DictWriter(
            self._trajectory_stream,
            fieldnames=(
                "execution_id",
                "run_id",
                "variant_id",
                "loop_indices",
                "trajectory_id",
                "trajectory_index",
                "rng_stream_seed",
                "trajectory_step",
                "point_id",
                "sequence",
                "latent_index",
                "adjusted_time_step",
            ),
        )
        self._local_enumeration_writer = csv.DictWriter(
            self._local_enumeration_stream,
            fieldnames=(
                "execution_id",
                "run_id",
                "variant_id",
                "loop_indices",
                "input_count",
                "input_latent_start",
                "input_latent_count",
                "input_sequences",
                "output_count",
                "output_sequences_sha256",
            ),
        )
        self._step_writer.writeheader()
        self._candidate_writer.writeheader()
        self._trajectory_writer.writeheader()
        self._local_enumeration_writer.writeheader()

    def begin_step(
        self,
        step: "Step",
        batch: "CandidateBatch",
        scope: ExecutionScope,
        context: "OptimizationContext",
    ) -> tuple[int, float, int, int]:
        with self._lock:
            execution_id = self._next_execution_id
            self._next_execution_id += 1
        return (
            execution_id,
            perf_counter(),
            context.state.oracle_calls,
            context.state.generated_candidates,
        )

    def end_step(
        self,
        handle: Any,
        step: "Step",
        input_batch: "CandidateBatch",
        output_batch: "CandidateBatch",
        scope: ExecutionScope,
        context: "OptimizationContext",
    ) -> None:
        from pep_compass.optimization.components.oracles.base import Oracle

        is_oracle = isinstance(step, Oracle)
        if self.level == "short" and not is_oracle:
            return
        step_row = {
            "execution_id": handle[0],
            "run_id": self.run_id,
            "variant_id": self.variant_id,
            "step_name": step.name,
            "path": "/".join(scope.path),
            "depth": scope.depth,
            "loop_indices": json.dumps(scope.loop_indices),
            "branch_names": json.dumps(scope.branch_names),
            "branch_indices": json.dumps(scope.branch_indices),
            "input_size": len(input_batch),
            "output_size": len(output_batch),
            "status": "completed",
            "duration_seconds": perf_counter() - handle[1],
            "oracle_calls_before": handle[2],
            "oracle_calls_after": context.state.oracle_calls,
            "generated_candidates_before": handle[3],
            "generated_candidates_after": context.state.generated_candidates,
            "error": "",
        }
        candidate_rows = []
        if self.level == "all" or is_oracle:
            candidate_rows = self._candidate_rows(handle[0], output_batch)
        with self._lock:
            trajectory_rows = (
                self._trajectory_rows(handle[0], output_batch, scope)
                if self.level in {"normal", "all"}
                and step.name == "SorbesWalker"
                else []
            )
            local_enumeration_row = (
                self._local_enumeration_row(
                    handle[0], input_batch, output_batch, scope
                )
                if self.level in {"normal", "all"}
                and step.name == "LocalEnumeration"
                else None
            )
            self._step_writer.writerow(step_row)
            self._candidate_writer.writerows(candidate_rows)
            self._trajectory_writer.writerows(trajectory_rows)
            if local_enumeration_row is not None:
                self._local_enumeration_writer.writerow(local_enumeration_row)
            self._step_stream.flush()
            self._candidate_stream.flush()
            self._trajectory_stream.flush()
            self._local_enumeration_stream.flush()

    def fail_step(self, handle, step, input_batch, scope, context, error):
        row = {
            "execution_id": handle[0],
            "run_id": self.run_id,
            "variant_id": self.variant_id,
            "step_name": step.name,
            "path": "/".join(scope.path),
            "depth": scope.depth,
            "loop_indices": json.dumps(scope.loop_indices),
            "branch_names": json.dumps(scope.branch_names),
            "branch_indices": json.dumps(scope.branch_indices),
            "input_size": len(input_batch),
            "output_size": 0,
            "status": "failed",
            "duration_seconds": perf_counter() - handle[1],
            "oracle_calls_before": handle[2],
            "oracle_calls_after": context.state.oracle_calls,
            "generated_candidates_before": handle[3],
            "generated_candidates_after": context.state.generated_candidates,
            "error": f"{type(error).__name__}: {error}",
        }
        with self._lock:
            self._step_writer.writerow(row)
            self._step_stream.flush()

    def _candidate_rows(
        self,
        execution_id: int,
        batch: "CandidateBatch",
    ) -> list[dict[str, Any]]:
        rows = []
        for index, sequence in enumerate(batch.sequences):
            latent_origin = ""
            if self.store_latents:
                latent_origin = json.dumps(
                    batch.latent_origins[index].detach().cpu().tolist(),
                    separators=(",", ":"),
                )
            rows.append(
                {
                    "execution_id": execution_id,
                    "candidate_index": index,
                    "sequence": sequence,
                    "latent_origin": latent_origin,
                    "fields": (
                        self._serialize_fields(batch, index)
                        if self.store_fields
                        else ""
                    ),
                }
            )
        return rows

    def _serialize_fields(self, batch: "CandidateBatch", index: int) -> str:
        from pep_compass.data.optimization import (
            ObjectField,
            OptionalField,
            SharedField,
            TensorField,
        )

        values: dict[str, Any] = {}
        for name, field_value in batch.fields.items():
            if self.field_names is not None and name not in self.field_names:
                continue
            if isinstance(field_value, TensorField):
                values[name] = field_value.values[index].detach().cpu().tolist()
            elif isinstance(field_value, ObjectField):
                values[name] = repr(field_value.values[index])
            elif isinstance(field_value, SharedField):
                values[name] = repr(field_value.value)
            elif isinstance(field_value, OptionalField):
                values[name] = {"valid": bool(field_value.valid[index].item())}
        return json.dumps(values, separators=(",", ":"))

    def _trajectory_rows(
        self,
        execution_id: int,
        batch: "CandidateBatch",
        scope: ExecutionScope,
    ) -> list[dict[str, Any]]:
        """Store compact SORBES checkpoints required for deterministic replay."""
        from pep_compass.data.optimization import ObjectField, TensorField

        object_names = ("tracking.trajectory_id", "point_id")
        tensor_names = (
            "tracking.trajectory_index",
            "tracking.rng_stream_seed",
            "tracking.trajectory_step",
            "walker.adjusted_time_step",
        )
        objects = {name: batch.fields.get(name) for name in object_names}
        tensors = {name: batch.fields.get(name) for name in tensor_names}
        if not isinstance(objects["tracking.trajectory_id"], ObjectField):
            return []
        latent_start = len(self._trajectory_latents)
        self._trajectory_latents.extend(batch.latent_origins.detach().cpu().unbind(0))
        rows = []
        for index, sequence in enumerate(batch.sequences):
            rows.append(
                {
                    "execution_id": execution_id,
                    "run_id": self.run_id,
                    "variant_id": self.variant_id,
                    "loop_indices": json.dumps(scope.loop_indices),
                    "trajectory_id": objects["tracking.trajectory_id"].values[index],
                    "trajectory_index": self._tensor_scalar(
                        tensors["tracking.trajectory_index"], index
                    ),
                    "rng_stream_seed": self._tensor_scalar(
                        tensors["tracking.rng_stream_seed"], index
                    ),
                    "trajectory_step": self._tensor_scalar(
                        tensors["tracking.trajectory_step"], index
                    ),
                    "point_id": (
                        objects["point_id"].values[index]
                        if isinstance(objects["point_id"], ObjectField)
                        else ""
                    ),
                    "sequence": sequence,
                    "latent_index": latent_start + index,
                    "adjusted_time_step": self._tensor_scalar(
                        tensors["walker.adjusted_time_step"], index
                    ),
                }
            )
        return rows

    def _local_enumeration_row(
        self,
        execution_id: int,
        input_batch: "CandidateBatch",
        output_batch: "CandidateBatch",
        scope: ExecutionScope,
    ) -> dict[str, Any]:
        """Checkpoint one local-enumeration boundary without retaining its batch."""
        latent_start = len(self._local_input_latents)
        self._local_input_latents.extend(
            input_batch.latent_origins.detach().cpu().unbind(0)
        )
        payload = json.dumps(
            list(output_batch.sequences),
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return {
            "execution_id": execution_id,
            "run_id": self.run_id,
            "variant_id": self.variant_id,
            "loop_indices": json.dumps(scope.loop_indices),
            "input_count": len(input_batch),
            "input_latent_start": latent_start,
            "input_latent_count": len(input_batch),
            "input_sequences": json.dumps(input_batch.sequences),
            "output_count": len(output_batch),
            "output_sequences_sha256": hashlib.sha256(payload).hexdigest(),
        }

    @staticmethod
    def _tensor_scalar(field: Any, index: int) -> int | float | str:
        """Return one scalar tensor field value or an empty CSV value."""
        from pep_compass.data.optimization import TensorField

        if not isinstance(field, TensorField):
            return ""
        return field.values[index].item()

    def close(self) -> None:
        with self._lock:
            self._step_stream.flush()
            self._candidate_stream.flush()
            self._trajectory_stream.flush()
            self._local_enumeration_stream.flush()
            trajectory_latents = (
                torch.stack(self._trajectory_latents)
                if self._trajectory_latents
                else torch.empty((0, 0))
            )
            local_input_latents = (
                torch.stack(self._local_input_latents)
                if self._local_input_latents
                else torch.empty((0, 0))
            )
            torch.save(
                trajectory_latents,
                self.output_directory / "trajectory_latents.pt",
            )
            torch.save(
                local_input_latents,
                self.output_directory / "local_enumeration_inputs.pt",
            )
            self._step_stream.close()
            self._candidate_stream.close()
            self._trajectory_stream.close()
            self._local_enumeration_stream.close()
