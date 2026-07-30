"""Base contract for every composable optimization operation."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pep_compass.optimization.batch import CandidateBatch
from pep_compass.optimization.context import OptimizationContext


class Step(ABC):
    """Transform a candidate batch while preserving the common data contract."""

    @property
    def name(self) -> str:
        """Return the stable tracking name of this step."""
        return self.__class__.__name__

    def precompute(self, context: OptimizationContext) -> None:
        """Prepare reusable state once before the optimization run.

        :param context: Runtime services shared by all steps.
        :type context: OptimizationContext
        """

    def prepare_iteration(
        self,
        batch: CandidateBatch,
        context: OptimizationContext,
    ) -> None:
        """Refresh state that depends on the current candidate batch."""

    def __call__(
        self,
        batch: CandidateBatch,
        context: OptimizationContext,
    ) -> CandidateBatch:
        """Execute the step with automatic, depth-aware tracking."""
        step_context = context.enter_step(self.name)
        self.prepare_iteration(batch, step_context)
        enabled = step_context.tracker.is_enabled(step_context.scope)
        handle = None
        if enabled:
            handle = step_context.tracker.begin_step(self, batch, step_context.scope)
        result = self._execute(batch, step_context)
        if enabled:
            step_context.tracker.end_step(
                handle,
                self,
                batch,
                result,
                step_context.scope,
            )
        return result

    @abstractmethod
    def _execute(
        self,
        batch: CandidateBatch,
        context: OptimizationContext,
    ) -> CandidateBatch:
        """Implement the batch transformation in a concrete step."""
