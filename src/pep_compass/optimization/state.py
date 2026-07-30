"""Mutable state shared by otherwise composable optimization steps."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OptimizationLimits:
    """Optional safety limits independent from the configured step tree."""

    oracle_calls: int | None = None
    generated_candidates: int | None = None


@dataclass
class OptimizationState:
    """Observations, counters, and stop state of one optimization run."""

    limits: OptimizationLimits = field(default_factory=OptimizationLimits)
    observations: dict[str, dict[str, float]] = field(default_factory=dict)
    oracle_calls: int = 0
    generated_candidates: int = 0
    stop_requested: bool = False

    def remaining_oracle_calls(self) -> int | None:
        """Return remaining oracle calls, or ``None`` for an unlimited run."""
        if self.limits.oracle_calls is None:
            return None
        return max(self.limits.oracle_calls - self.oracle_calls, 0)

    def record_observations(
        self,
        objective: str,
        sequences: tuple[str, ...],
        scores: list[float],
    ) -> None:
        """Store latest objective values and update the oracle-call counter."""
        objective_observations = self.observations.setdefault(objective, {})
        objective_observations.update(zip(sequences, scores))
        self.oracle_calls += len(sequences)
        if (
            self.limits.oracle_calls is not None
            and self.oracle_calls >= self.limits.oracle_calls
        ):
            self.stop_requested = True

    def record_generated_candidates(self, count: int) -> None:
        """Update generated-candidate count and its optional safety limit."""
        self.generated_candidates += count
        if (
            self.limits.generated_candidates is not None
            and self.generated_candidates >= self.limits.generated_candidates
        ):
            self.stop_requested = True
