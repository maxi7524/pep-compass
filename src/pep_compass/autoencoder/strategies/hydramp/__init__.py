"""HydrAMP autoencoder method and named-model registrations."""

from __future__ import annotations

from typing import Any

import torch

from pep_compass.autoencoder.registry import (
    AutoencoderModelDescriptor,
    AutoencoderRegistry,
)
from pep_compass.utils.strategy_factory import parameter_contract


@parameter_contract(
    accepted={
        "device",
        "jacobian_mode",
        "default_condition",
        "temp",
        "jacobian_eps",
        "field_eps",
        "model_name",
    },
    required={"jacobian_eps", "field_eps"},
)
def build_hydramp(*, device: str = "cpu", **parameters: Any):
    """Build HydrAMP and normalize its configured condition tensor."""
    from pep_compass.autoencoder.strategies.hydramp.adapter import (
        HydrampAutoencoder,
    )

    if "default_condition" in parameters:
        parameters["default_condition"] = torch.as_tensor(
            parameters["default_condition"],
            device=device,
        )
    return HydrampAutoencoder(device=device, **parameters)


def register_hydramp() -> None:
    """Register the HydrAMP adapter and bundled named model variants."""
    if "hydramp" not in AutoencoderRegistry.methods():
        AutoencoderRegistry.register_method("hydramp")(build_hydramp)
    if "article_25" not in AutoencoderRegistry.models("hydramp"):
        AutoencoderRegistry.register_model(
            "hydramp",
            AutoencoderModelDescriptor(
                "article_25",
                {"model_name": "article_25"},
            ),
        )


__all__ = ["build_hydramp", "register_hydramp"]
