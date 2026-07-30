"""Sequential, iterative, and branching optimization compositions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

from pep_compass.optimization.batch import CandidateBatch
from pep_compass.optimization.context import OptimizationContext
from pep_compass.optimization.step import Step
from pep_compass.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)

ParallelExecution = Literal["sequential", "concurrent"]
MergeMethod = Literal["concatenate", "interleave", "select_best", "weighted_sample"]


class BatchMerger(ABC):
    """Merge outputs from logically parallel optimization branches."""

    @abstractmethod
    def __call__(self, batches: Sequence[CandidateBatch]) -> CandidateBatch:
        """Return one candidate batch containing merged branch outputs."""


class ConcatenateMerger(BatchMerger):
    """Append branch outputs without deduplication, ordering, or selection."""

    def __call__(self, batches: Sequence[CandidateBatch]) -> CandidateBatch:
        return CandidateBatch.concatenate(batches)


def interleave_batches(batches: Sequence[CandidateBatch]) -> CandidateBatch:
    """Alternate candidates from branches while preserving branch order.

    This future merge policy should take one candidate from every non-empty
    branch in turn. It is useful when downstream truncation must not favor the
    branch placed first in configuration.
    """
    # TODO: implement a fair round-robin merge with aligned field propagation.
    raise NotImplementedError("The interleave merge policy is not implemented yet.")


def select_best_batches(batches: Sequence[CandidateBatch]) -> CandidateBatch:
    """Merge branches by a configured score and retain the best candidates.

    The implementation will require an explicit score field, direction, and
    output size. Selection must remain separate from deduplication.
    """
    # TODO: implement score-field validation, stable ranking, and top-k selection.
    raise NotImplementedError("The select_best merge policy is not implemented yet.")


def weighted_sample_batches(batches: Sequence[CandidateBatch]) -> CandidateBatch:
    """Sample a configured proportion of candidates from every branch.

    The implementation will require deterministic branch weights, a requested
    output size, and RNG supplied by :class:`OptimizationContext`.
    """
    # TODO: implement deterministic weighted sampling without implicit deduplication.
    raise NotImplementedError(
        "The weighted_sample merge policy is not implemented yet."
    )


@dataclass(frozen=True)
class FunctionMerger(BatchMerger):
    """Expose a merge function through the common merger interface."""

    function: object

    def __call__(self, batches: Sequence[CandidateBatch]) -> CandidateBatch:
        return self.function(batches)  # type: ignore[operator]


def build_merger(method: MergeMethod) -> BatchMerger:
    """Build a branch merger from its configuration name."""
    if method == "concatenate":
        return ConcatenateMerger()
    if method == "interleave":
        return FunctionMerger(interleave_batches)
    if method == "select_best":
        return FunctionMerger(select_best_batches)
    if method == "weighted_sample":
        return FunctionMerger(weighted_sample_batches)
    raise ValueError(f"Unsupported merge method: {method}")


class Flow(Step):
    """Execute child steps sequentially in their configured order."""

    def __init__(self, steps: Sequence[Step]) -> None:
        self.steps = tuple(steps)

    def precompute(self, context: OptimizationContext) -> None:
        for step in self.steps:
            step.precompute(context.enter_step(step.name))

    def _execute(
        self,
        batch: CandidateBatch,
        context: OptimizationContext,
    ) -> CandidateBatch:
        result = batch
        for step in self.steps:
            result = step(result, context)
        return result


class Loop(Step):
    """Execute one child step a fixed number of times."""

    def __init__(self, body: Step, iterations: int) -> None:
        if iterations < 0:
            raise ValueError("Loop iterations cannot be negative.")
        self.body = body
        self.iterations = iterations

    def precompute(self, context: OptimizationContext) -> None:
        self.body.precompute(context.enter_step(self.body.name))

    def _execute(
        self,
        batch: CandidateBatch,
        context: OptimizationContext,
    ) -> CandidateBatch:
        result = batch
        for index in range(self.iterations):
            if context.state.stop_requested:
                break
            result = self.body(result, context.enter_iteration(index))
        return result


class Parallel(Step):
    """Run independent branches from one input and merge their output batches."""

    def __init__(
        self,
        branches: Mapping[str, Step],
        *,
        execution: ParallelExecution = "sequential",
        merger: BatchMerger | None = None,
    ) -> None:
        if not branches:
            raise ValueError("Parallel requires at least one branch.")
        if execution not in {"sequential", "concurrent"}:
            raise ValueError("Parallel execution must be sequential or concurrent.")
        self.branches = dict(branches)
        self.execution = execution
        self.merger = merger or ConcatenateMerger()

    def precompute(self, context: OptimizationContext) -> None:
        for index, (name, branch) in enumerate(self.branches.items()):
            branch.precompute(context.enter_branch(name, index))

    def _execute(
        self,
        batch: CandidateBatch,
        context: OptimizationContext,
    ) -> CandidateBatch:
        branch_items = list(self.branches.items())
        if self.execution == "sequential":
            outputs = [
                branch(batch, context.enter_branch(name, index))
                for index, (name, branch) in enumerate(branch_items)
            ]
        else:
            logger.debug(
                "Executing %s optimization branches concurrently.", len(branch_items)
            )
            with ThreadPoolExecutor(max_workers=len(branch_items)) as executor:
                futures = [
                    executor.submit(branch, batch, context.enter_branch(name, index))
                    for index, (name, branch) in enumerate(branch_items)
                ]
                outputs = [future.result() for future in futures]
        return self.merger(outputs)
