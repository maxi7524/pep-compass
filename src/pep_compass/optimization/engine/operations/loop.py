"""Repeated traversal of one configured child flow.

``PipelineBuilder`` wraps YAML ``loop.steps`` in a ``Flow`` and installs it as
the loop body. Each iteration receives the previous iteration's output and a
scope containing the iteration index.
"""

from pep_compass.data.optimization import CandidateBatch
from pep_compass.optimization.engine.execution.context import OptimizationContext
from pep_compass.optimization.engine.execution.step import Step


class Loop(Step):
    """Execute one child step a fixed number of times."""

    def __init__(self, body: Step, iterations: int) -> None:
        if iterations < 0:
            raise ValueError("Loop iterations cannot be negative.")
        self.body = body
        self.iterations = iterations

    def precompute(self, context: OptimizationContext) -> None:
        """Precompute the repeated body once."""
        self.body.precompute(context.enter_step(self.body.name))

    def _execute(self, batch: CandidateBatch, context: OptimizationContext) -> CandidateBatch:
        """Execute iterations until completion or a runtime stop condition."""
        result = batch
        for index in range(self.iterations):
            if context.state.stop_requested or len(result) == 0:
                break
            result = self.body(result, context.enter_iteration(index))
            context.state.record_iteration()
        return result
