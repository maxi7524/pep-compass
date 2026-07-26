"""Regression tests for LE-BO optimization control flow."""

from contextlib import nullcontext

from pep_compass.optimization.lebo.local_enumeration_bayesian_optimizer import (
    LocalEnumerationBayesianOptimizer,
)


def test_bayesian_optimization_stops_when_candidate_pool_is_empty() -> None:
    """Return a stop signal when local enumeration yields no test peptide."""
    optimizer = object.__new__(LocalEnumerationBayesianOptimizer)
    optimizer.scored_peptides = {"AAAA": 0.0, "AAAC": 1.0}
    optimizer.timer = lambda _: nullcontext()
    optimizer._turbo_filter = lambda: []

    assert optimizer._bayesian_optimization(max_scorer_calls=2) is None
    assert optimizer.iteration_evaluations == []
