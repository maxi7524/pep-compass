"""Sequential traversal of configured child steps.

``PipelineBuilder`` creates a ``Flow`` for every YAML ``steps`` list. During
execution it passes each returned ``CandidateBatch`` directly to the next
child. Every child enters the standard ``Step.__call__`` lifecycle.
"""

from collections.abc import Sequence

from pep_compass.data_schemas.candidates import CandidateBatch
from pep_compass.optimization.engine.execution.context import OptimizationContext
from pep_compass.steps.base import Step


class Flow(Step):
    """Execute child steps sequentially in their configured order."""

    category = "flow"
    kind = "sequence"

    def __init__(self, steps: Sequence[Step]) -> None:
        self._configure({})
        self.steps = tuple(steps)
        if any(not isinstance(step, Step) for step in self.steps):
            raise TypeError("Flow requires Step components.")

    def graph_children(self):
        return {"steps": self.steps}

    def precompute(self, context: OptimizationContext) -> None:
        """Precompute every child in execution order."""
        for step in self.steps:
            step.precompute(context.enter_step(step.name))

    def _execute(
        self, batch: CandidateBatch, context: OptimizationContext
    ) -> CandidateBatch:
        """Pass each intermediate batch to the next child step."""
        result = batch
        for step in self.steps:
            result = step(result, context)
        return result
