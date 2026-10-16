"""Lifecycle contract for every executable pipeline node.

``PipelineBuilder`` creates a tree whose composite and leaf nodes all inherit
from :class:`Step`. ``PepCompassPipeline.run`` invokes its root. The public
``__call__`` method supplies iteration preparation, tracking, resource
sampling, timing, and failure reporting before delegating only the actual
transformation to :meth:`_execute`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import wraps
from time import perf_counter

from pep_compass.data_schemas.candidates import CandidateBatch
from pep_compass.optimization.engine.execution.context import OptimizationContext
from pep_compass.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


class Step(ABC):
    """Transform a candidate batch within the common execution lifecycle.

    Subclasses normally implement only
    * :meth:`_execute`.
    Override
    * :meth:`precompute` for state independent of the current batch,
    * :meth:`prepare_call` for state refreshed immediately before each call.

    Subclasses **must not** override :meth:`__call__`, because
    doing so would bypass tracking, memory sampling, timing, and error events.

    Composite subclasses are
    * ``Flow``,
    * ``Loop``,
    * ``Parallel``
    under ``optimization.engine.operations``. Leaf subclasses are component-family
    base classes under ``optimization.components``.
    """

    def __init_subclass__(cls, **kwargs):
        """Keep lifecycle orchestration central, including preparation failures."""
        super().__init_subclass__(**kwargs)
        if "__call__" in cls.__dict__:
            raise TypeError("Executable steps implement _execute, not __call__.")
        preparation = cls.__dict__.get("precompute")
        if preparation is not None:

            @wraps(preparation)
            def tracked_precompute(self, context):
                scoped = context.enter_node(getattr(self, "node_id", context.node_id))
                started = perf_counter()
                error = None
                try:
                    return preparation(self, scoped)
                except Exception as caught:
                    error = caught
                    raise
                finally:
                    if scoped.saver is not None:
                        scoped.saver.record_precompute(
                            self, scoped, perf_counter() - started, error
                        )

            cls.precompute = tracked_precompute

    @property
    def name(self) -> str:
        """Return the stable tracking name of this step."""
        return getattr(self, "display_name", None) or self.__class__.__name__

    def display_info(self) -> dict:
        """Return configured parameters selected for compact graph presentation.

        Components may override this method; it must not perform computation.

        :return: Small mapping of meaningful parameter names and values.
        """
        configuration = getattr(self, "configuration", {})
        selected = getattr(self, "display_parameter_names", ())
        return {key: configuration[key] for key in selected if key in configuration}

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

    def prepare_call(self, batch: CandidateBatch, context: OptimizationContext) -> None:
        """Prepare each invocation, including calls outside loop iterations.

        :param batch: Incoming candidate batch.
        :param context: Scoped services and fresh run state.
        """
        self.prepare_iteration(batch, context)

    def __call__(
        self,
        batch: CandidateBatch,
        context: OptimizationContext,
    ) -> CandidateBatch:
        """Apply preparation, transformation and observation as one lifecycle.

        :param batch: Aligned input candidates; implementations do not mutate it.
        :param context: Scoped run services and state.
        :return: Aligned output batch.
        :raises Exception: Original preparation, computation or recording error.
        """
        # Invocation scope and before-step observations
        step_context = context.enter_step(self.name).enter_node(
            getattr(self, "node_id", context.node_id)
        )
        step_context.stability_monitor.sample(
            f"step.before:{'/'.join(step_context.scope.path)}",
            batch,
            step_context.state,
        )
        enabled = step_context.tracker.is_enabled(step_context.scope)
        handle = None
        if enabled:
            handle = step_context.tracker.begin_step(
                self, batch, step_context.scope, step_context
            )
        started_at = perf_counter()

        # Preparation, transformation and completion share failure reporting
        try:
            self.prepare_call(batch, step_context)
            if step_context.saver is not None:
                step_context.saver.execution_started(handle)
            result = self._execute(batch, step_context)
            if not isinstance(result, CandidateBatch):
                raise TypeError(
                    f"{self.name} must return CandidateBatch, got {type(result).__name__}."
                )
            if enabled:
                step_context.tracker.end_step(
                    handle, self, batch, result, step_context.scope, step_context
                )
            step_context.stability_monitor.sample(
                f"step.after:{'/'.join(step_context.scope.path)}",
                result,
                step_context.state,
            )
        except Exception as error:
            if enabled:
                try:
                    step_context.tracker.fail_step(
                        handle, self, batch, step_context.scope, step_context, error
                    )
                except Exception:
                    logger.error(
                        "Failed to record the original step error path=%s.",
                        step_context.scope.path,
                        exc_info=True,
                    )
            raise

        # Compact diagnostics never retain candidate batches
        logger.debug(
            "Step path=%s input_candidates=%s output_candidates=%s duration_seconds=%.6f.",
            "/".join(step_context.scope.path),
            len(batch),
            len(result),
            perf_counter() - started_at,
        )
        return result

    @abstractmethod
    def _execute(
        self,
        batch: CandidateBatch,
        context: OptimizationContext,
    ) -> CandidateBatch:
        """Implement the batch transformation in a concrete step."""
