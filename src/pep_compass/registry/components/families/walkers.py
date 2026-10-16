"""Family registration and construction through the shared registry contract."""

from pep_compass.optimization.components.walkers.base import Walker
from pep_compass.registry.components.catalog import component_catalog
from pep_compass.registry.components.manager import ComponentManager
from pep_compass.registry.factories.registry import Registry


class WalkerManager(ComponentManager):
    """Register and construct walker implementations."""

    registry = Registry("walker", expected_type=Walker)
    services = frozenset({"autoencoder"})


component_catalog.register_family("walker", WalkerManager.registry)
