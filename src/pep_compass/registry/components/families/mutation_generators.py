"""Family registration and construction through the shared registry contract."""

from pep_compass.optimization.components.mutation_generators.base import (
    MutationGenerator,
)
from pep_compass.registry.components.catalog import component_catalog
from pep_compass.registry.components.manager import ComponentManager
from pep_compass.registry.factories.registry import Registry


class MutationGeneratorManager(ComponentManager):
    """Register and construct mutation_generator implementations."""

    registry = Registry("mutation_generator", expected_type=MutationGenerator)
    services = frozenset({"autoencoder"})


component_catalog.register_family(
    "mutation_generator", MutationGeneratorManager.registry
)
