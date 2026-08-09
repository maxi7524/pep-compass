"""Parallel optimization-branch composition."""

from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping
from typing import Literal

from pep_compass.data.optimization import CandidateBatch
from pep_compass.optimization.engine.context import OptimizationContext
from pep_compass.optimization.engine.merging import BatchMerger, ConcatenateMerger
from pep_compass.optimization.engine.step import Step
from pep_compass.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)
ParallelExecution = Literal["auto", "sequential", "concurrent"]


class Parallel(Step):
    """Run independent branches from one input and merge their outputs."""

    def __init__(
        self,
        branches: Mapping[str, Step],
        *,
        execution: ParallelExecution = "auto",
        merger: BatchMerger | None = None,
    ) -> None:
        if not branches:
            raise ValueError("Parallel requires at least one branch.")
        if execution not in {"auto", "sequential", "concurrent"}:
            raise ValueError("Parallel execution must be auto, sequential, or concurrent.")
        self.branches = dict(branches)
        self.execution = execution
        self.merger = merger or ConcatenateMerger()

    def precompute(self, context: OptimizationContext) -> None:
        """Precompute every branch with its independent execution scope."""
        for index, (name, branch) in enumerate(self.branches.items()):
            branch.precompute(context.enter_branch(name, index))

    def _execute(self, batch: CandidateBatch, context: OptimizationContext) -> CandidateBatch:
        """Execute branches sequentially or concurrently and merge outputs."""
        branch_items = list(self.branches.items())
        if self.execution == "sequential":
            outputs = [
                branch(batch, context.enter_branch(name, index))
                for index, (name, branch) in enumerate(branch_items)
            ]
        else:
            logger.debug("Executing %s optimization branches concurrently.", len(branch_items))
            with ThreadPoolExecutor(max_workers=len(branch_items)) as executor:
                futures = [
                    executor.submit(branch, batch, context.enter_branch(name, index))
                    for index, (name, branch) in enumerate(branch_items)
                ]
                outputs = [future.result() for future in futures]
        return self.merger(outputs)
