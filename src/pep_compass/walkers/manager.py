"""Walker strategy registry."""

from typing import Callable

from pep_compass.walkers.base import Walker
from pep_compass.utils.strategy_factory import build_with_services


class WalkerManager:
    """Register and construct latent-space walker strategies."""

    _registry: dict[str, Callable[..., Walker]] = {}

    @classmethod
    def register(
        cls, name: str
    ) -> Callable[[Callable[..., Walker]], Callable[..., Walker]]:
        """Register a walker factory under a configuration name."""

        def decorator(factory: Callable[..., Walker]) -> Callable[..., Walker]:
            cls._registry[name] = factory
            return factory

        return decorator

    @classmethod
    def build(cls, method: str, *, services=None, **parameters) -> Walker:
        """Construct a registered walker strategy."""
        try:
            factory = cls._registry[method]
        except KeyError as error:
            raise ValueError(f"Unknown walker method: {method}") from error
        return build_with_services(factory, parameters, services)

    @classmethod
    def methods(cls) -> tuple[str, ...]:
        """Return registered walker method names."""
        return tuple(sorted(cls._registry))
