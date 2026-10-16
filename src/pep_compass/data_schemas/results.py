"""Shared computation and run-result contracts with lazy storage references."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from pep_compass.data_schemas.candidates import Candidate, CandidateBatch

if TYPE_CHECKING:
    from pep_compass.optimization.saving.reader import AnalysisReader


@dataclass(frozen=True)
class OptimizationResult:
    """Evaluated candidates and an optional objective summary.

    :param candidates: Archive for the last evaluated objective, or terminal batch
        when no observations were recorded; retained for compatibility.
    :param terminal_candidates: Exact final batch propagated by the computation.
    :param evaluated_archives: Per-objective archives of completed observations.
    """

    candidates: CandidateBatch
    best_candidate: Candidate | None = None
    best_score: float | None = None
    objective_name: str | None = None
    objective_direction: str | None = None
    terminal_candidates: CandidateBatch | None = None
    evaluated_archives: Mapping[str, CandidateBatch] = field(default_factory=dict)


@dataclass(frozen=True)
class RunResult:
    """One completed run and a lazy view over exactly its retained information."""

    summary: dict
    reader: AnalysisReader
    path: Path | None

    @property
    def archive(self):
        """Return the base archive of evaluated peptides."""
        return self.reader.get_archive()

    def save(self, path):
        """Export only retained information as settings.json and results.sqlite."""
        return self.reader.backend.export(path)

    def close(self):
        """Release lazy result resources after all reads are complete."""
        self.reader.close()
