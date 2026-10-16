"""Execution of named branches from one shared input batch.

``PipelineBuilder`` converts configured branches or replicas into child
``Step`` objects. This operation executes every child against the same input,
then delegates output combination to a merger from ``parallel.merge``.
"""

from concurrent.futures import ThreadPoolExecutor
from threading import Condition
from typing import Literal

from pep_compass.data_schemas.candidates import CandidateBatch
from pep_compass.optimization.engine.execution.context import OptimizationContext
from pep_compass.steps.base import Step
from pep_compass.utils.logger import get_custom_logger
from pep_compass.utils.randomness import random_scope

logger = get_custom_logger(__name__)
ParallelExecution = Literal["sequential", "concurrent"]


class Parallel(Step):
    """Run independent branches from one input and merge their outputs."""

    display_parameters = ("execution",)
    category = "flow"
    kind = "parallel"

    def __init__(self, branches, *, execution="sequential"):
        from pep_compass.steps.flow.strategies.sequence import Flow

        if not branches:
            raise ValueError("Parallel requires at least one branch.")
        if execution not in {"sequential", "concurrent"}:
            raise ValueError("Parallel execution must be sequential or concurrent.")
        self.branches = {
            key: value
            if isinstance(value, Flow)
            else Flow([value] if isinstance(value, Step) else value)
            for key, value in branches.items()
        }
        self.execution = execution
        self._configure(dict(execution=execution))

    def graph_children(self):
        return {"branches": {name: flow.steps for name, flow in self.branches.items()}}

    def precompute(self, context: OptimizationContext) -> None:
        """Precompute every branch with its independent execution scope."""
        for index, (name, branch) in enumerate(self.branches.items()):
            branch.precompute(context.enter_branch(name, index))

    def _execute(
        self, batch: CandidateBatch, context: OptimizationContext
    ) -> CandidateBatch:
        """Execute branches sequentially or concurrently and merge outputs."""
        branch_items = list(self.branches.items())
        contexts = [
            context.enter_branch(name, index)
            for index, (name, _) in enumerate(branch_items)
        ]

        def execute(index):
            """Execute one branch with an isolated deterministic RNG scope."""
            with random_scope(contexts[index].seed, serialize=False):
                return branch_items[index][1](batch, contexts[index])

        if self.execution == "sequential":
            outputs = [execute(index) for index in range(len(branch_items))]
        else:
            # Ordered access to shared run resources
            ## REMARK: Threads share model adapters, tracking and OptimizationState.
            ## Order their critical sections to preserve RNG streams, IDs, budgets
            ## and trust regions. Use runtime subprocesses for independent parallelism.
            logger.info(
                "Ordering %s concurrent branches sharing mutable runtime services.",
                len(branch_items),
            )
            condition = Condition()
            next_branch = [0]

            def ordered_execute(index):
                with condition:
                    condition.wait_for(lambda: next_branch[0] == index)
                    try:
                        return execute(index)
                    finally:
                        next_branch[0] += 1
                        condition.notify_all()

            with ThreadPoolExecutor(max_workers=len(branch_items)) as executor:
                futures = [
                    executor.submit(ordered_execute, index)
                    for index in range(len(branch_items))
                ]
                outputs = [future.result() for future in futures]
        return CandidateBatch.concatenate(outputs)
