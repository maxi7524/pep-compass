"""Tracking contracts for composable optimization steps."""

from __future__ import annotations

from abc import ABC
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from pep_compass.optimization.batch import CandidateBatch
    from pep_compass.optimization.step import Step
    from pep_compass.optimization.context import OptimizationContext

TrackingLevel = Literal["short", "normal", "all"]


@dataclass(frozen=True)
class ExecutionScope:
    """Location of a step execution in a nested optimization tree."""

    path: tuple[str, ...] = ()
    loop_indices: tuple[int, ...] = ()
    branch_names: tuple[str, ...] = ()
    branch_indices: tuple[int, ...] = ()

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
            branch_indices=self.branch_indices,
        )

    def loop_iteration(self, index: int) -> "ExecutionScope":
        """Return a scope for one nested loop iteration."""
        return ExecutionScope(
            path=(*self.path, f"iteration[{index}]"),
            loop_indices=(*self.loop_indices, index),
            branch_names=self.branch_names,
            branch_indices=self.branch_indices,
        )

    def branch(self, name: str, index: int) -> "ExecutionScope":
        """Return a scope for one logical parallel branch."""
        return ExecutionScope(
            path=(*self.path, f"branch[{name}]"),
            loop_indices=self.loop_indices,
            branch_names=(*self.branch_names, name),
            branch_indices=(*self.branch_indices, index),
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
        context: "OptimizationContext",
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
        context: "OptimizationContext",
    ) -> None:
        """Finish one tracked execution."""

    def fail_step(
        self,
        handle: Any,
        step: "Step",
        input_batch: "CandidateBatch",
        scope: ExecutionScope,
        context: "OptimizationContext",
        error: BaseException,
    ) -> None:
        """Record a failed step before its exception propagates."""

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
    branch_indices: tuple[int, ...]
    input_size: int
    output_size: int
    status: str = "completed"
    duration_seconds: float = 0.0
    oracle_calls_before: int = 0
    oracle_calls_after: int = 0
    generated_candidates_before: int = 0
    generated_candidates_after: int = 0
    error: str | None = None


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

    def begin_step(self, step, batch, scope, context):
        return (
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
        record = StepExecutionRecord(
            step_name=step.name,
            path=scope.path,
            loop_indices=scope.loop_indices,
            branch_names=scope.branch_names,
            branch_indices=scope.branch_indices,
            input_size=len(input_batch),
            output_size=len(output_batch),
            duration_seconds=perf_counter() - handle[0],
            oracle_calls_before=handle[1],
            oracle_calls_after=context.state.oracle_calls,
            generated_candidates_before=handle[2],
            generated_candidates_after=context.state.generated_candidates,
        )
        with self._lock:
            self.records.append(record)

    def fail_step(self, handle, step, input_batch, scope, context, error):
        record = StepExecutionRecord(
            step_name=step.name,
            path=scope.path,
            loop_indices=scope.loop_indices,
            branch_names=scope.branch_names,
            branch_indices=scope.branch_indices,
            input_size=len(input_batch),
            output_size=0,
            status="failed",
            duration_seconds=perf_counter() - handle[0],
            oracle_calls_before=handle[1],
            oracle_calls_after=context.state.oracle_calls,
            generated_candidates_before=handle[2],
            generated_candidates_after=context.state.generated_candidates,
            error=f"{type(error).__name__}: {error}",
        )
        with self._lock:
            self.records.append(record)


class CSVStepTracker(StepTracker):
    """Stream depth-aware optimization records to normalized CSV files.

    ``field_names`` limits serialized candidate fields when field storage is
    enabled. Step summaries are flushed after each execution so interrupted
    runs retain their last completed or failed step.
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
        self._step_writer.writeheader()
        self._candidate_writer.writeheader()

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
        from pep_compass.oracles.base import Oracle

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
            self._step_writer.writerow(step_row)
            self._candidate_writer.writerows(candidate_rows)
            self._step_stream.flush()
            self._candidate_stream.flush()

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
        from pep_compass.optimization.batch import (
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

    def close(self) -> None:
        with self._lock:
            self._step_stream.flush()
            self._candidate_stream.flush()
            self._step_stream.close()
            self._candidate_stream.close()
