"""Lightweight declaration of one shared HydrAMP model resource."""

from pep_compass.configuration.objects import Configurable
from pep_compass.utils.validation import validate_positive


class HydrAMP(Configurable):
    """Configure the shared encoder; load weights only during build.

    :param model: Model subdirectory containing encoder and decoder weights.
    :param device: PyTorch computation device.
    :param jacobian_mode: Exact autograd or finite-difference Jacobians.
    :param jacobian_eps: Positive finite-difference increment.
    :param field_eps: Positive derivative increment for decoder fields.
    :param temp: Positive decoder temperature.
    :param default_condition: Two decoder conditioning values.
    """

    def __init__(
        self,
        *,
        model="article_25",
        device="cpu",
        jacobian_mode="strict",
        jacobian_eps=0.001,
        field_eps=0.001,
        temp=1.0,
        default_condition=None,
    ):
        validate_positive(jacobian_eps, "HydrAMP.jacobian_eps")
        validate_positive(field_eps, "HydrAMP.field_eps")
        validate_positive(temp, "HydrAMP.temp")
        if jacobian_mode not in {"strict", "approx"}:
            raise ValueError("Invalid Jacobian mode.")
        condition = [1.0, 1.0] if default_condition is None else list(default_condition)
        if len(condition) != 2:
            raise ValueError("HydrAMP requires two conditioning values.")
        self._configure(
            dict(
                model=model,
                device=str(device),
                jacobian_mode=jacobian_mode,
                jacobian_eps=jacobian_eps,
                field_eps=field_eps,
                temp=temp,
                default_condition=condition,
            )
        )

    def build(self, *, device=None):
        """Initialize the numerical model using this immutable constructor snapshot."""
        import torch

        from pep_compass.autoencoder.strategies.hydramp.adapter import (
            HydrampAutoencoder,
        )

        values = self.configuration
        values["model_name"] = values.pop("model")
        values["device"] = device or values["device"]
        values["default_condition"] = torch.tensor(
            values["default_condition"], device=values["device"]
        )
        return HydrampAutoencoder(**values)
