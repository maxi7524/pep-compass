"""Uniform family facade; strategy-specific behavior belongs to registrations."""

from pep_compass.registry.factories.registry import Registry


class ComponentManager:
    """Delegate registration, discovery and construction to one typed registry."""

    registry: Registry
    services: frozenset[str] = frozenset()

    @classmethod
    def register(cls, name, **contract):
        """Return the family's registration decorator with default dependencies."""
        contract.setdefault("services", cls.services)
        return cls.registry.register(name, **contract)

    @classmethod
    def methods(cls):
        """Return available strategy names without constructing components."""
        return cls.registry.names()

    @classmethod
    def describe(cls, method):
        """Return structured strategy documentation."""
        return cls.registry.describe(method)

    @classmethod
    def validate(cls, method, parameters):
        """Validate parameters through their registered definition."""
        cls.registry.validate(method, parameters)

    @classmethod
    def build(cls, method, *, services=None, **parameters):
        """Resolve a strategy using the shared construction contract."""
        return cls.registry.resolve(method, parameters, services=services)
