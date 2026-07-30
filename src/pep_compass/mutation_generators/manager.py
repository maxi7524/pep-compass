"""Mutation generator strategy registry."""

from typing import Callable

from pep_compass.mutation_generators.base import MutationGenerator


class MutationGeneratorManager:
    """Register and construct mutation generator strategies."""

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
    def build(cls, method: str, **parameters) -> MutationGenerator:
        """Construct a registered mutation generator."""
        try:
            factory = cls._registry[method]
        except KeyError as error:
            raise ValueError(f"Unknown mutation generator method: {method}") from error
        return factory(**parameters)
