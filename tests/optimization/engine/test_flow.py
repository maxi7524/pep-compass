"""Tests for composable engine graph operations."""

from pep_compass.optimization.engine.context import OptimizationContext
from pep_compass.optimization.engine import Flow, Loop, Parallel
from pep_compass.optimization.tracking import InMemoryStepTracker
from fixtures.autoencoders import MockAutoencoder
from fixtures.candidates import candidate_batch
from fixtures.components import SuffixStep


def test_parallel_merges_outputs_in_declared_branch_order() -> None:
    """Concurrent completion order must not affect merged candidate order."""
    parallel = Parallel(
        {"first": SuffixStep("1"), "second": SuffixStep("2")},
        execution="concurrent",
    )

    result = parallel(
        candidate_batch(),
        OptimizationContext(MockAutoencoder(), seed=7),
    )

    assert result.sequences == ("AA1", "BB1", "AA2", "BB2")


def test_loop_stops_when_a_step_produces_an_empty_batch() -> None:
    """Empty candidate sets must not execute redundant loop iterations."""
    class EmptyStep(SuffixStep):
        calls = 0

        def _execute(self, batch, context):
            self.calls += 1
            return batch.select([])

    step = EmptyStep("")

    result = Loop(step, iterations=3)(
        candidate_batch(),
        OptimizationContext(MockAutoencoder()),
    )

    assert len(result) == 0
    assert step.calls == 1


def test_nested_tracking_preserves_loop_and_branch_identity() -> None:
    """Tracker records must identify replicated work independently."""
    tracker = InMemoryStepTracker()
    graph = Loop(
        Parallel({"left": Flow([SuffixStep("L")])}),
        iterations=2,
    )

    graph(
        candidate_batch(),
        OptimizationContext(MockAutoencoder(), tracker=tracker),
    )

    records = [record for record in tracker.records if record.step_name == "SuffixStep"]
    assert [record.loop_indices for record in records] == [(0,), (1,)]
    assert [record.branch_names for record in records] == [("left",), ("left",)]
    assert [record.branch_indices for record in records] == [(0,), (0,)]
