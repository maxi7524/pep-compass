"""Runtime context shared by optimization steps."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from pep_compass.optimization.tracking import (
    ExecutionScope,
    NullStepTracker,
    StepTracker,
)
from pep_compass.optimization.state import OptimizationState


@dataclass(frozen=True)
class OptimizationContext:
    """Immutable execution services and scope for an optimization run."""

    encoder_decoder: Any
    tracker: StepTracker = field(default_factory=NullStepTracker)
    scope: ExecutionScope = field(default_factory=ExecutionScope)
    seed: int | None = None
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    state: OptimizationState = field(default_factory=OptimizationState)

    def enter_step(self, name: str) -> "OptimizationContext":
        """Return a context nested below a named step."""
        return replace(self, scope=self.scope.child(name))

    def enter_iteration(self, index: int) -> "OptimizationContext":
        """Return a context for one loop iteration."""
        return replace(self, scope=self.scope.loop_iteration(index))

    def enter_branch(self, name: str, index: int) -> "OptimizationContext":
        """Return an independently seeded context for a parallel branch."""
        branch_seed = None if self.seed is None else self.seed + index + 1
        return replace(
            self,
            scope=self.scope.branch(name),
            seed=branch_seed,
            rng=np.random.default_rng(branch_seed),
        )
