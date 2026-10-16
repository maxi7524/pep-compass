"""Built-in autoencoder factories and named model declarations."""

from typing import Any

import torch

from pep_compass.autoencoder.strategies.hydramp.configuration import (
    HydrampConfig,
    canonical_parameters,
)
from pep_compass.registry.components.families.autoencoders import AutoencoderRegistry
from pep_compass.registry.models.definitions import (
    AutoencoderModelDescriptor,
    FileArtifact,
)


@AutoencoderRegistry.register_method(
    "hydramp",
    config_type=HydrampConfig,
    normalizer=canonical_parameters,
    display_parameters=("model_name", "jacobian_mode", "temp"),
)
def build_hydramp(*, device: str = "cpu", **parameters: Any):
    """Build HydrAMP and normalize its configured condition tensor."""
    from pep_compass.autoencoder.strategies.hydramp.adapter import (
        HydrampAutoencoder,
    )

    if parameters.get("default_condition") is not None:
        parameters["default_condition"] = torch.as_tensor(
            parameters["default_condition"],
            device=device,
        )
    return HydrampAutoencoder(device=device, **parameters)


AutoencoderRegistry.register_model(
    "hydramp",
    AutoencoderModelDescriptor(
        "article_25",
        {"model_name": "article_25"},
        directory="article_25",
        artifacts=(
            FileArtifact("encoder_weights.pickle"),
            FileArtifact("decoder_weights.pickle"),
        ),
    ),
)
