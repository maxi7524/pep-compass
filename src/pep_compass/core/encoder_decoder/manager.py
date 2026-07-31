"""Encoder-decoder strategy registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class EncoderDecoderManager:
    """Register and construct encoder-decoder strategies for the composition root."""

    _registry: dict[str, Callable[..., Any]] = {}

    @classmethod
    def register(cls, name: str):
        """Register an encoder-decoder factory under a configuration name."""

        def decorator(factory):
            cls._registry[name] = factory
            return factory

        return decorator

    @classmethod
    def build(cls, method: str, **parameters):
        """Construct a registered encoder-decoder strategy."""
        try:
            factory = cls._registry[method]
        except KeyError as error:
            raise ValueError(f"Unknown encoder-decoder method: {method}") from error
        return factory(**parameters)

    @classmethod
    def methods(cls) -> tuple[str, ...]:
        """Return registered encoder-decoder method names."""
        return tuple(sorted(cls._registry))

    @classmethod
    def factory(cls, method: str) -> Callable[..., Any]:
        """Return a registered factory without constructing its model."""
        try:
            return cls._registry[method]
        except KeyError as error:
            raise ValueError(f"Unknown encoder-decoder method: {method}") from error
