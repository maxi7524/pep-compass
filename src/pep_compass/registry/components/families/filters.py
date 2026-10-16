"""Family registration and construction through the shared registry contract."""

from pep_compass.optimization.components.filters.base import Filter
from pep_compass.registry.components.catalog import component_catalog
from pep_compass.registry.components.manager import ComponentManager
from pep_compass.registry.factories.registry import Registry


class FilterManager(ComponentManager):
    """Register and construct filter implementations."""

    registry = Registry("filter", expected_type=Filter)
    services = frozenset({"autoencoder"})


component_catalog.register_family("filter", FilterManager.registry)
