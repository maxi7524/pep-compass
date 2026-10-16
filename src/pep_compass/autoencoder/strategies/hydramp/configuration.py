"""Inspectable parameter defaults for the named HydrAMP model adapter."""

from dataclasses import dataclass, field

from pep_compass.utils.validation import validate_positive


@dataclass(frozen=True, slots=True)
class HydrampConfig:
    """Separate named model selection from numerical differentiation settings."""

    model_name: str
    jacobian_eps: float = field(
        default=0.001,
        metadata={
            "description": "Positive forward-difference increment for approximate Jacobians."
        },
    )
    field_eps: float = field(
        default=0.001,
        metadata={"description": "Positive increment for decoder field derivatives."},
    )
    jacobian_mode: str = "strict"
    temp: float = 1.0
    default_condition: object = field(default_factory=lambda: [1.0, 1.0])
    device: object = "cpu"

    def __post_init__(self):
        validate_positive(self.jacobian_eps, "hydramp.jacobian_eps")
        validate_positive(self.field_eps, "hydramp.field_eps")
        validate_positive(self.temp, "hydramp.temp")
        if self.jacobian_mode not in {"strict", "approx"}:
            raise ValueError("HydrAMP jacobian_mode must be strict or approx.")
        if self.default_condition is not None and len(self.default_condition) != 2:
            raise ValueError("HydrAMP default_condition requires two entries.")


def canonical_parameters(values):
    """Normalize device and tensor conditions to portable configuration values."""
    result = dict(values)
    result["device"] = str(result["device"])
    condition = result["default_condition"]
    if condition is None:
        condition = [1.0, 1.0]
    if hasattr(condition, "detach"):
        condition = condition.detach().cpu().tolist()
    result["default_condition"] = list(condition)
    return result
