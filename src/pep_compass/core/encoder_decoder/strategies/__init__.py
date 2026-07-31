"""Built-in encoder-decoder strategies."""

from typing import Any

import torch

from pep_compass.core.encoder_decoder.manager import EncoderDecoderManager


@EncoderDecoderManager.register("hydramp")
def build_hydramp(*, device: str = "cpu", **parameters: Any):
    """Build HydrAMP while converting its configured condition to a tensor."""
    from pep_compass.core.encoder_decoder.strategies.hydramp.adapter import (
        HydrAMPEncoderDecoder,
    )

    if "default_condition" in parameters:
        parameters["default_condition"] = torch.as_tensor(
            parameters["default_condition"], device=device
        )
    return HydrAMPEncoderDecoder(device=device, **parameters)
