"""Execution of materialized plans through a selected runtime workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

from pep_compass.optimization.stability_estimation.monitoring import (
    MemorySnapshot,
    StabilityMonitor,
)
from pep_compass.optimization.tracking import InMemoryStepTracker, StepExecutionRecord
from pep_compass.runtime.configuration.schema import RuntimeConfiguration
from pep_compass.runtime.output.tracking import CSVStepTracker
from pep_compass.runtime.output.writer import ResultWriter
from pep_compass.runtime.planning.plan import ExecutionPlan, PlannedRun
from pep_compass.runtime.workflows.base import RuntimeWorkflow
from pep_compass.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


@dataclass(frozen=True, slots=True)
class RunExecution:
    """Record the terminal status of one executed plan entry."""

    entry: PlannedRun
    status: Literal["completed", "failed", "skipped"]
    error: str | None = None
    duration_seconds: float = 0.0
    candidate_count: int | None = None
    step_records: tuple[StepExecutionRecord, ...] = ()
    memory_snapshots: tuple[MemorySnapshot, ...] = ()


class RuntimeRunner:
    """Execute planned runs and attach runtime-owned output implementations.

    The runner builds ``CSVStepTracker`` only when a ``ResultWriter`` provides
    a run directory. Without output it injects ``NullStepTracker``. It does not
    decide when intermediate batches are released: operations replace local
    references as batches move through the step tree, and trackers serialize
    rows without retaining those batches.
    """

    def __init__(
        self,
        configuration: RuntimeConfiguration,
        workflow: RuntimeWorkflow,
        writer: ResultWriter | None,
        *,
        capture_diagnostics: bool = False,
    ) -> None:
        self.configuration = configuration
        self.workflow = workflow
        self.writer = writer
        self.capture_diagnostics = capture_diagnostics

    def run(self, plan: ExecutionPlan) -> tuple[RunExecution, ...]:
        """Execute plan entries sequentially in the current process."""
        executions = []
        for entry in plan.entries:
            execution = self._run_entry(entry)
            executions.append(execution)
            if execution.status == "failed" and not self.configuration.execution.continue_on_error:
                break
        return tuple(executions)

    def _run_entry(self, entry: PlannedRun) -> RunExecution:
        """Build, execute and persist one plan entry."""
        if self.writer is not None and self.configuration.execution.resume and self.writer.is_completed(entry):
            logger.info("Skipping completed run %s.", entry.run_id)
            return RunExecution(entry, "skipped")
        run_directory = self.writer.run_directory(entry) if self.writer is not None else None
        tracker = self._build_tracker(entry, run_directory)
        monitor = StabilityMonitor(
            enabled=self.configuration.tracking.monitor_stability,
            detailed_every_sample=self.capture_diagnostics,
        )
        if self.writer is not None:
            self.writer.write_running(entry)
        try:
            started_at = perf_counter()
            pipeline = self.workflow.build_pipeline(
                entry.variant.pipeline,
                tracker=tracker,
                stability_monitor=monitor,
            )
            result = pipeline.run([entry.task.sequence], seed=entry.seed)
            if self.writer is not None:
                self.writer.write_completed(entry, result, monitor.snapshots)
            return RunExecution(
                entry,
                "completed",
                duration_seconds=perf_counter() - started_at,
                candidate_count=len(result.candidates),
                step_records=tuple(getattr(tracker, "records", ())),
                memory_snapshots=tuple(monitor.snapshots),
            )
        except Exception as error:
            logger.error("Run %s failed: %s", entry.run_id, error, exc_info=True)
            if self.writer is not None:
                self.writer.write_failed(entry, error)
            return RunExecution(
                entry,
                "failed",
                f"{type(error).__name__}: {error}",
                duration_seconds=(
                    perf_counter() - started_at if "started_at" in locals() else 0.0
                ),
                step_records=tuple(getattr(tracker, "records", ())),
                memory_snapshots=tuple(monitor.snapshots),
            )

    def _build_tracker(self, entry: PlannedRun, directory: Path | None):
        """Construct a CSV tracker or a no-output tracker for one run."""
        if directory is None:
            if self.capture_diagnostics:
                return InMemoryStepTracker(
                    level="normal",
                    max_depth=self.configuration.tracking.max_depth,
                )
            from pep_compass.optimization.tracking import NullStepTracker

            return NullStepTracker()
        return CSVStepTracker(
            directory / "tracking",
            level=self.configuration.tracking.level,
            max_depth=self.configuration.tracking.max_depth,
            store_latents=self.configuration.tracking.store_latents,
            store_fields=self.configuration.tracking.store_fields,
            field_names=self.configuration.tracking.field_names,
            run_id=entry.run_id,
            variant_id=entry.variant.variant_id,
        )
