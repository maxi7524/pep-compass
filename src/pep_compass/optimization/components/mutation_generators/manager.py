"""Mutation generator strategy registry."""

from typing import Callable

from pep_compass.optimization.components.mutation_generators.base import MutationGenerator
from pep_compass.utils.strategy_factory import build_with_services, validate_factory_parameters


class MutationGeneratorManager:
    """Map configured generator names to factories used by ``PipelineBuilder``."""

    _registry: dict[str, Callable[..., MutationGenerator]] = {}

    @classmethod
    def register(
        cls,
        name: str,
    ) -> Callable[[Callable[..., MutationGenerator]], Callable[..., MutationGenerator]]:
        """Register a mutation generator factory."""

        def decorator(
            factory: Callable[..., MutationGenerator],
        ) -> Callable[..., MutationGenerator]:
            cls._registry[name] = factory
            return factory

        return decorator

    @classmethod
    def build(cls, method: str, *, services=None, **parameters) -> MutationGenerator:
        """Construct a registered mutation generator."""
        try:
            factory = cls._registry[method]
        except KeyError as error:
            raise ValueError(f"Unknown mutation generator method: {method}") from error
        return build_with_services(factory, parameters, services)

    @classmethod
    def methods(cls) -> tuple[str, ...]:
        """Return registered mutation-generator method names."""
        return tuple(sorted(cls._registry))

    @classmethod
    def validate(cls, method: str, parameters) -> None:
        """Validate a declaration without constructing the generator."""
        try:
            factory = cls._registry[method]
        except KeyError as error:
            raise ValueError(f"Unknown mutation generator method: {method}") from error
        validate_factory_parameters(factory, parameters, service_names={"autoencoder"})
