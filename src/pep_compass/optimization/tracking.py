"""Tracking contracts for composable optimization steps."""

from __future__ import annotations

from abc import ABC
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from pep_compass.optimization.batch import CandidateBatch
    from pep_compass.optimization.step import Step

TrackingLevel = Literal["short", "normal", "all"]


@dataclass(frozen=True)
class ExecutionScope:
    """Location of a step execution in a nested optimization tree."""

    path: tuple[str, ...] = ()
    loop_indices: tuple[int, ...] = ()
    branch_names: tuple[str, ...] = ()

    @property
    def depth(self) -> int:
        """Return the nesting depth of the current execution."""
        return len(self.path)

    def child(self, name: str) -> "ExecutionScope":
        """Return a scope nested below a named step."""
        return ExecutionScope(
            path=(*self.path, name),
            loop_indices=self.loop_indices,
            branch_names=self.branch_names,
        )

    def loop_iteration(self, index: int) -> "ExecutionScope":
        """Return a scope for one nested loop iteration."""
        return ExecutionScope(
            path=(*self.path, f"iteration[{index}]"),
            loop_indices=(*self.loop_indices, index),
            branch_names=self.branch_names,
        )

    def branch(self, name: str) -> "ExecutionScope":
        """Return a scope for one logical parallel branch."""
        return ExecutionScope(
            path=(*self.path, f"branch[{name}]"),
            loop_indices=self.loop_indices,
            branch_names=(*self.branch_names, name),
        )


class StepTracker(ABC):
    """Lifecycle sink invoked around every optimization step."""

    level: TrackingLevel = "short"
    max_depth: int | None = None

    def is_enabled(self, scope: ExecutionScope) -> bool:
        """Return whether data should be collected for a scope."""
        return self.max_depth is None or scope.depth <= self.max_depth

    def begin_step(
        self,
        step: "Step",
        batch: "CandidateBatch",
        scope: ExecutionScope,
    ) -> Any:
        """Start one tracked execution and return an opaque handle."""
        return None

    def end_step(
        self,
        handle: Any,
        step: "Step",
        input_batch: "CandidateBatch",
        output_batch: "CandidateBatch",
        scope: ExecutionScope,
    ) -> None:
        """Finish one tracked execution."""

    def close(self) -> None:
        """Flush and close tracker resources after one optimization run."""


class NullStepTracker(StepTracker):
    """No-op tracker used when detailed tracking is disabled."""

    def is_enabled(self, scope: ExecutionScope) -> bool:
        return False


@dataclass(frozen=True)
class StepExecutionRecord:
    """Compact record of one completed step execution."""

    step_name: str
    path: tuple[str, ...]
    loop_indices: tuple[int, ...]
    branch_names: tuple[str, ...]
    input_size: int
    output_size: int


class InMemoryStepTracker(StepTracker):
    """Thread-safe tracker used by tests and lightweight debugging runs."""

    def __init__(
        self,
        *,
        level: TrackingLevel = "normal",
        max_depth: int | None = None,
    ) -> None:
        self.level = level
        self.max_depth = max_depth
        self.records: list[StepExecutionRecord] = []
        self._lock = Lock()

    def end_step(
        self,
        handle: Any,
        step: "Step",
        input_batch: "CandidateBatch",
        output_batch: "CandidateBatch",
        scope: ExecutionScope,
    ) -> None:
        record = StepExecutionRecord(
            step_name=step.name,
            path=scope.path,
            loop_indices=scope.loop_indices,
            branch_names=scope.branch_names,
            input_size=len(input_batch),
            output_size=len(output_batch),
        )
        with self._lock:
            self.records.append(record)


class CSVStepTracker(StepTracker):
    """Stream depth-aware optimization records to normalized CSV files."""

    def __init__(
        self,
        output_directory: str | Path,
        *,
        level: TrackingLevel = "normal",
        max_depth: int | None = None,
        store_latents: bool = False,
        store_fields: bool = False,
    ) -> None:
        if level not in {"short", "normal", "all"}:
            raise ValueError("Tracking level must be short, normal, or all.")
        self.level = level
        self.max_depth = max_depth
        self.store_latents = store_latents
        self.store_fields = store_fields
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
        self._step_writer = csv.DictWriter(
            self._step_stream,
            fieldnames=(
                "execution_id",
                "step_name",
                "path",
                "depth",
                "loop_indices",
                "branch_names",
                "input_size",
                "output_size",
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
        self._step_writer.writeheader()
        self._candidate_writer.writeheader()

    def begin_step(
        self,
        step: "Step",
        batch: "CandidateBatch",
        scope: ExecutionScope,
    ) -> int:
        with self._lock:
            execution_id = self._next_execution_id
            self._next_execution_id += 1
        return execution_id

    def end_step(
        self,
        handle: Any,
        step: "Step",
        input_batch: "CandidateBatch",
        output_batch: "CandidateBatch",
        scope: ExecutionScope,
    ) -> None:
        from pep_compass.oracles.base import Oracle

        is_oracle = isinstance(step, Oracle)
        if self.level == "short" and not is_oracle:
            return
        step_row = {
            "execution_id": handle,
            "step_name": step.name,
            "path": "/".join(scope.path),
            "depth": scope.depth,
            "loop_indices": json.dumps(scope.loop_indices),
            "branch_names": json.dumps(scope.branch_names),
            "input_size": len(input_batch),
            "output_size": len(output_batch),
        }
        candidate_rows = []
        if self.level == "all" or is_oracle:
            candidate_rows = self._candidate_rows(handle, output_batch)
        with self._lock:
            self._step_writer.writerow(step_row)
            self._candidate_writer.writerows(candidate_rows)

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

    @staticmethod
    def _serialize_fields(batch: "CandidateBatch", index: int) -> str:
        from pep_compass.optimization.batch import (
            ObjectField,
            OptionalField,
            SharedField,
            TensorField,
        )

        values: dict[str, Any] = {}
        for name, field_value in batch.fields.items():
            if isinstance(field_value, TensorField):
                values[name] = field_value.values[index].detach().cpu().tolist()
            elif isinstance(field_value, ObjectField):
                values[name] = repr(field_value.values[index])
            elif isinstance(field_value, SharedField):
                values[name] = repr(field_value.value)
            elif isinstance(field_value, OptionalField):
                values[name] = {"valid": bool(field_value.valid[index].item())}
        return json.dumps(values, separators=(",", ":"))

    def close(self) -> None:
        with self._lock:
            self._step_stream.flush()
            self._candidate_stream.flush()
            self._step_stream.close()
            self._candidate_stream.close()
