"""Filter strategy registry."""

from typing import Callable

from pep_compass.filters.base import Filter
from pep_compass.utils.strategy_factory import build_with_services


class FilterManager:
    """Register and construct filter strategies."""

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
