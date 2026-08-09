"""Tests for the public manually constructed PepCompass pipeline."""

from pep_compass.optimization.engine.flow import Flow
from pep_compass.optimization.pipeline import PepCompassPipeline
from fixtures.autoencoders import MockAutoencoder
from fixtures.components import SuffixStep


def test_pipeline_can_be_constructed_without_core_or_runtime() -> None:
    """The computation API must remain independent from YAML and planning."""
    pipeline = PepCompassPipeline(
        autoencoder=MockAutoencoder(),
        root=Flow([SuffixStep("X")]),
    )

    result = pipeline.run(["AA"], seed=7)

    assert result.candidates.sequences == ("AAX",)
    assert result.candidates.latent_origins.shape == (1, 2)
