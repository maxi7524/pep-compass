"""Filter strategy registry."""

from typing import Callable

from pep_compass.optimization.components.filters.base import Filter
from pep_compass.utils.strategy_factory import build_with_services, validate_factory_parameters


class FilterManager:
    """Map configured filter names to factories used by ``PipelineBuilder``."""

    _registry: dict[str, Callable[..., Filter]] = {}

    @classmethod
    def register(
        cls, name: str
    ) -> Callable[[Callable[..., Filter]], Callable[..., Filter]]:
        """Register a filter factory."""

        def decorator(factory: Callable[..., Filter]) -> Callable[..., Filter]:
            cls._registry[name] = factory
            return factory

        return decorator

    @classmethod
    def build(cls, method: str, *, services=None, **parameters) -> Filter:
        """Construct a registered filter strategy."""
        try:
            factory = cls._registry[method]
        except KeyError as error:
            raise ValueError(f"Unknown filter method: {method}") from error
        return build_with_services(factory, parameters, services)

    @classmethod
    def methods(cls) -> tuple[str, ...]:
        """Return registered filter method names."""
        return tuple(sorted(cls._registry))

    @classmethod
    def validate(cls, method: str, parameters) -> None:
        """Validate a strategy declaration without constructing the filter."""
        try:
            factory = cls._registry[method]
        except KeyError as error:
            raise ValueError(f"Unknown filter method: {method}") from error
        validate_factory_parameters(factory, parameters, service_names={"autoencoder"})
