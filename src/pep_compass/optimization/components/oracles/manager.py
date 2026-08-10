"""Oracle strategy registry."""

from typing import Callable

from pep_compass.optimization.components.oracles.base import Oracle
from pep_compass.utils.strategy_factory import validate_factory_parameters


class OracleManager:
    """Map configured oracle names to lazy factories used by ``PipelineBuilder``."""

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

    @classmethod
    def validate(cls, method: str, parameters) -> None:
        """Validate a declaration without loading an oracle model."""
        try:
            factory = cls._registry[method]
        except KeyError as error:
            raise ValueError(f"Unknown oracle method: {method}") from error
        validate_factory_parameters(factory, parameters)
