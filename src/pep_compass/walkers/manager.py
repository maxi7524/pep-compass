"""Walker strategy registry."""

from typing import Callable

from pep_compass.walkers.base import Walker


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
    def build(cls, method: str, **parameters) -> Walker:
        """Construct a registered walker strategy."""
        try:
            factory = cls._registry[method]
        except KeyError as error:
            raise ValueError(f"Unknown walker method: {method}") from error
        return factory(**parameters)
