"""Tracking contracts for composable optimization steps."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
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
