"""Runtime services propagated through the executable step tree.

``PepCompassPipeline.run`` creates the root context. ``Step.__call__`` and the
composite operations derive scoped copies as execution enters a step,
iteration, or branch. All copies intentionally share ``OptimizationState``;
only their path and branch-local random generator differ.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from pep_compass.optimization.engine.execution.state import OptimizationState
from pep_compass.optimization.stability_estimation.monitoring import (
    NullStabilityMonitor,
    StabilityMonitor,
)
from pep_compass.optimization.tracking import (
    ExecutionScope,
    NullStepTracker,
    StepTracker,
)


@dataclass(frozen=True)
class OptimizationContext:
    """Carry services, scope, RNG, counters, and monitors through execution.

    The context does not decide which component runs next. That order is
    encoded in the ``Step`` tree built by ``PipelineBuilder``. ``Flow``,
    ``Loop``, and ``Parallel`` traverse that tree and pass derived contexts to
    their children.
    """

    autoencoder: Any
    tracker: StepTracker = field(default_factory=NullStepTracker)
    scope: ExecutionScope = field(default_factory=ExecutionScope)
    seed: int | None = None
    rng: np.random.Generator | None = None
    state: OptimizationState = field(default_factory=OptimizationState)
    stability_monitor: StabilityMonitor | NullStabilityMonitor = field(
        default_factory=NullStabilityMonitor
    )
    saver: Any = None
    node_id: str | None = None
    step_state: dict[str, Any] = field(default_factory=dict)

    def enter_node(self, node_id: str | None) -> "OptimizationContext":
        """Return a scope bound to a stable declaration identity."""
        return replace(self, node_id=node_id)

    def iteration_completed(self, index: int, *, trajectories: int = 1) -> None:
        """Emit one operator iteration independently of trajectory counters."""
        if self.saver is not None:
            self.saver.iteration_completed(self, index, trajectories)

    def iteration_started(self, index: int) -> None:
        """Mark one operator iteration without changing computational counters."""
        if self.saver is not None:
            self.saver.iteration_started(self, index)

    def __post_init__(self) -> None:
        """Create a seeded local stream when callers do not provide one."""
        if self.rng is None:
            object.__setattr__(self, "rng", np.random.default_rng(self.seed))

    def enter_step(self, name: str) -> "OptimizationContext":
        """Return a context nested below a named step."""
        return replace(self, scope=self.scope.child(name))

    def enter_iteration(self, index: int) -> "OptimizationContext":
        """Return a context for one loop iteration."""
        return replace(self, scope=self.scope.loop_iteration(index))

    def enter_branch(self, name: str, index: int) -> "OptimizationContext":
        """Return an independently seeded context for a parallel branch."""
        # A consumed parent-stream token distinguishes repeated branch calls;
        # structural scope separates nested branches and loop iterations.
        token = int(self.rng.integers(0, 2**63))
        payload = json.dumps(
            (
                self.seed,
                self.scope.path,
                self.scope.loop_indices,
                self.scope.branch_names,
                name,
                index,
                token,
            )
        ).encode()
        branch_seed = (
            None
            if self.seed is None
            else int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)
        )
        return replace(
            self,
            scope=self.scope.branch(name, index),
            seed=branch_seed,
            rng=np.random.default_rng(branch_seed),
        )
