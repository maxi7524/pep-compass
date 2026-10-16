"""Family registration and construction through the shared registry contract."""

from pep_compass.optimization.components.oracles.base import Oracle
from pep_compass.registry.components.catalog import component_catalog
from pep_compass.registry.components.manager import ComponentManager
from pep_compass.registry.factories.registry import Registry


class OracleManager(ComponentManager):
    """Register and construct oracle implementations."""

    registry = Registry("oracle", expected_type=Oracle)


component_catalog.register_family("oracle", OracleManager.registry)
