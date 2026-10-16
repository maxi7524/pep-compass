"""Deterministic registered resources shared by orchestration workflow tests."""

import torch

from pep_compass.autoencoder.base import Autoencoder
from pep_compass.core.wrapper import Pipeline, Step
from pep_compass.registry.components.families.autoencoders import AutoencoderRegistry
from pep_compass.registry.models.definitions import AutoencoderModelDescriptor
from pep_compass.registry.pipelines.catalog import register


@AutoencoderRegistry.register_method("fixture_encoder")
class FixtureAutoencoder(Autoencoder):
    """Two-dimensional encoder without files, downloads or random weights."""

    latent_dim = 2
    ambient_dim = 2

    def __init__(self, device="cpu"):
        super().__init__(jacobian_eps=0.01, field_eps=0.01)
        self.device = device

    def encode_peptides(self, peptides):
        return torch.tensor(
            [[float(len(sequence)), 1.0] for sequence in peptides], device=self.device
        )  # (B, 2)

    def decode_peptides(self, latent_positions):
        return ["AA" for _ in range(len(latent_positions))]

    def decoder_forward(self, latent_positions):
        return latent_positions  # (B, 2)

    def encoder_forward(self, encoded_sequences):
        return encoded_sequences  # (B, 2)


AutoencoderRegistry.register_model(
    "fixture_encoder", AutoencoderModelDescriptor("default", {})
)


@register("fixture")
def create(*, seed=7, count=2, device="cpu"):
    """Return a tiny pipeline that evaluates retained input peptides.

    :param seed: Reproducible RNG seed.
    :param count: Number of selected peptides before scoring.
    :param device: CPU resource device.
    :return: Editable pipeline with default oracle_calls=3.
    """
    pipeline = Pipeline.create(
        steps=[
            Step.create(
                component="filter",
                strategy="candidate_subset",
                parameters={"count": count},
            ),
            Step.create(component="oracle", strategy="hydrophobicity"),
        ]
    )
    pipeline.add_autoencoder("fixture_encoder", model="default", device=device)
    pipeline.set_runner_settings(execution={"seed": seed, "oracle_calls": 3})
    return pipeline
