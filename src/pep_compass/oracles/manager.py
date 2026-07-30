"""Oracle strategy registry."""

from typing import Callable

from pep_compass.oracles.base import Oracle


class OracleManager:
    """Register and construct objective evaluation strategies."""

    _registry: dict[str, Callable[..., Oracle]] = {}

    @classmethod
    def register(
        cls, name: str
    ) -> Callable[[Callable[..., Oracle]], Callable[..., Oracle]]:
        """Register an oracle factory."""

        def decorator(factory: Callable[..., Oracle]) -> Callable[..., Oracle]:
            cls._registry[name] = factory
            return factory

        return decorator

    @classmethod
    def build(cls, method: str, **parameters) -> Oracle:
        """Construct a registered oracle strategy."""
        try:
            factory = cls._registry[method]
        except KeyError as error:
            raise ValueError(f"Unknown oracle method: {method}") from error
        return factory(**parameters)

    @classmethod
    def methods(cls) -> tuple[str, ...]:
        """Return registered method names in deterministic order.

        :return: Registered oracle method names.
        :rtype: tuple[str, ...]
        """
        return tuple(sorted(cls._registry))
