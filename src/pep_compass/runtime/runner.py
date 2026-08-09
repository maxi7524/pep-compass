"""Execution of materialized plans through a selected runtime workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pep_compass.optimization.stability_estimation.monitoring import StabilityMonitor
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
    ) -> None:
        self.configuration = configuration
        self.workflow = workflow
        self.writer = writer

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
        monitor = StabilityMonitor(enabled=self.configuration.tracking.monitor_stability)
        if self.writer is not None:
            self.writer.write_running(entry)
        try:
            pipeline = self.workflow.build_pipeline(
                entry.variant.pipeline,
                tracker=tracker,
                stability_monitor=monitor,
            )
            result = pipeline.run([entry.task.sequence], seed=entry.seed)
            if self.writer is not None:
                self.writer.write_completed(entry, result, monitor.snapshots)
            return RunExecution(entry, "completed")
        except Exception as error:
            logger.error("Run %s failed: %s", entry.run_id, error, exc_info=True)
            if self.writer is not None:
                self.writer.write_failed(entry, error)
            return RunExecution(entry, "failed", f"{type(error).__name__}: {error}")

    def _build_tracker(self, entry: PlannedRun, directory: Path | None):
        """Construct a CSV tracker or a no-output tracker for one run."""
        if directory is None:
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
