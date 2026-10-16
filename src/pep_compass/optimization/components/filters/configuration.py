"""Typed public defaults for composed filters with forwarding factories."""

from dataclasses import dataclass
from typing import Literal

from pep_compass.utils.validation import validate_integer


@dataclass(frozen=True, slots=True)
class DeduplicateConfig:
    """Candidate identity used by deterministic first-occurrence selection."""

    key: Literal["sequence", "sequence_and_latent"] = "sequence"


@dataclass(frozen=True, slots=True)
class CandidateSubsetConfig:
    """Bound output size using random selection or a declared score field."""

    count: int
    mode: Literal["random", "highest", "lowest"] = "random"
    score_field: str | None = None

    def __post_init__(self):
        validate_integer(self.count, "candidate_subset.count", 1)
        if self.mode != "random" and not self.score_field:
            raise ValueError("Ranked candidate subset modes require score_field.")


@dataclass(frozen=True, slots=True)
class MutationPoolConfig:
    """Bound mutation scoring independently of the run's generation budget."""

    maximum_candidates: int = 30000
    alphabet: list[str] | None = None


@dataclass(frozen=True, slots=True)
class RankedConfig:
    """Explicit score and selection strategy declarations."""

    scoring: dict
    selection: dict


@dataclass(frozen=True, slots=True)
class NucleusPoolConfig(MutationPoolConfig):
    """Select the high-probability part of a scored mutation pool."""

    top_p: float = 0.9
    temperature: float = 1.0


@dataclass(frozen=True, slots=True)
class MoveConfig(NucleusPoolConfig):
    """MOVE's published default nucleus mass differs from other pool filters."""

    top_p: float = 0.6


@dataclass(frozen=True, slots=True)
class TandemConfig(NucleusPoolConfig):
    """TANDEM additionally selects active decoder directions."""

    horizontal_threshold: float = 0.1


@dataclass(frozen=True, slots=True)
class LamsConfig(MutationPoolConfig):
    """LAMS similarity threshold and decoder direction cutoff."""

    similarity_threshold: float = 0.15
    horizontal_threshold: float = 0.1


@dataclass(frozen=True, slots=True)
class RandomPoolConfig(MutationPoolConfig):
    """Random controls use the same defaults for both supported modes."""

    selection_fraction: float = 0.6
    temperature: float = 1.0
    maximum_positions: int = 5
    residues_per_position: int = 4
