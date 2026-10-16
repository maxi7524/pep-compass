"""Tests for declarative construction of PepCompass pipelines."""

from fixtures.autoencoders import MockAutoencoder

from pep_compass.core.builder import PipelineBuilder
from pep_compass.data_schemas.specification import (
    FlowSpecification,
    PipelineSpecification,
)
from pep_compass.registry.components.catalog import ComponentCatalog


def test_builder_returns_executable_pep_compass_pipeline() -> None:
    """Core must construct, but not execute, a declared pipeline."""
    pipeline = PipelineBuilder(MockAutoencoder(), catalog=ComponentCatalog()).build(
        PipelineSpecification(FlowSpecification(()))
    )

    result = pipeline.run(["AA"])

    assert result.candidates.sequences == ("AA",)
