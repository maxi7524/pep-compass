"""Built-in encoder-decoder strategies."""

from typing import Any

import torch

from pep_compass.encoder_decoder.manager import EncoderDecoderManager
from pep_compass.utils.strategy_factory import parameter_contract


@EncoderDecoderManager.register("hydramp")
@parameter_contract(
    accepted={"device", "jacobian_mode", "default_condition", "temp", "jacobian_eps", "field_eps"},
    required={"jacobian_eps", "field_eps"},
)
def build_hydramp(*, device: str = "cpu", **parameters: Any):
    """Build HydrAMP while converting its configured condition to a tensor."""
    from pep_compass.encoder_decoder.strategies.hydramp.adapter import (
        HydrAMPEncoderDecoder,
    )

    if "default_condition" in parameters:
        parameters["default_condition"] = torch.as_tensor(
            parameters["default_condition"], device=device
        )
    return HydrAMPEncoderDecoder(device=device, **parameters)
