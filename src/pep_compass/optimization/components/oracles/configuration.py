"""Canonical lightweight defaults for lazy black-box model factories."""

import math
from dataclasses import dataclass, field

from pep_compass.utils.validation import validate_integer


@dataclass(frozen=True, slots=True)
class OracleConfig:
    """Keep adapter batching separate from the run's shared oracle budget."""

    batch_size: int | None = None
    parallelize: bool = False
    num_workers: int | None = None
    evaluation_budget: int | float | None = field(
        default=None,
        metadata={
            "description": "Optional underlying black-box budget; None is unlimited. The Runner budget is configured separately."
        },
    )
    force_isolation: bool = False
    evaluation_batch_size: int | None = field(
        default=None,
        metadata={
            "description": "Optional adapter microbatch size; never truncates a completed evaluation batch."
        },
    )

    def __post_init__(self):
        for name in ("batch_size", "num_workers", "evaluation_batch_size"):
            validate_integer(getattr(self, name), f"oracle.{name}", 1, optional=True)
        if self.evaluation_budget is not None and (
            isinstance(self.evaluation_budget, bool)
            or math.isnan(self.evaluation_budget)
            or self.evaluation_budget < 0
        ):
            raise ValueError("oracle.evaluation_budget must be non-negative or None.")


@dataclass(frozen=True, slots=True)
class ModelOracleConfig(OracleConfig):
    """Select registered weights without importing their implementation."""

    model: str = "default"
    models_directory: str | None = None


@dataclass(frozen=True, slots=True)
class DeviceModelOracleConfig(ModelOracleConfig):
    """Model-backed black box with an explicit device."""

    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class ApexConfig(DeviceModelOracleConfig):
    """APEX bacterial subset and ensemble aggregation."""

    mic_aggregate: str = "mean"
    mic_bacteria: str | list[int] = "all"


@dataclass(frozen=True, slots=True)
class EipredConfig(ModelOracleConfig):
    """EI-pred aggregation settings."""

    mic_aggregate: str = "mean"


@dataclass(frozen=True, slots=True)
class HydrophobicityConfig(OracleConfig):
    """Registered residue hydrophobicity scale."""

    scale: str = "eisenberg"


def canonical_parameters(values):
    """Represent an unlimited legacy budget using portable JSON null."""
    result = dict(values)
    if result.get("evaluation_budget") == math.inf:
        result["evaluation_budget"] = None
    return result
