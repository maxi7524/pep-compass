"""Registry of autoencoder methods and named model variants."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pep_compass.autoencoder.base import Autoencoder


AutoencoderFactoryCallable = Callable[..., Autoencoder]


@dataclass(frozen=True, slots=True)
class AutoencoderModelDescriptor:
    """Describe a named model variant supported by one autoencoder method.

    :param name: Public model name used in configuration.
    :param parameters: Loader parameters associated with the model variant.
    """

    name: str
    parameters: dict[str, Any]


class AutoencoderRegistry:
    """Store autoencoder factories and named model descriptors."""

    _methods: dict[str, AutoencoderFactoryCallable] = {}
    _models: dict[str, dict[str, AutoencoderModelDescriptor]] = {}

    @classmethod
    def register_method(
        cls,
        name: str,
    ) -> Callable[[AutoencoderFactoryCallable], AutoencoderFactoryCallable]:
        """Return a decorator registering one autoencoder implementation."""
        if not name:
            raise ValueError("Autoencoder method name cannot be empty.")

        def decorator(factory: AutoencoderFactoryCallable) -> AutoencoderFactoryCallable:
            if name in cls._methods:
                raise ValueError(f"Autoencoder method is already registered: {name}")
            cls._methods[name] = factory
            return factory

        return decorator

    @classmethod
    def register_model(
        cls,
        method: str,
        descriptor: AutoencoderModelDescriptor,
    ) -> None:
        """Register a named model variant for an autoencoder method."""
        models = cls._models.setdefault(method, {})
        if descriptor.name in models:
            raise ValueError(
                f"Autoencoder model is already registered: {method}/{descriptor.name}"
            )
        models[descriptor.name] = descriptor

    @classmethod
    def method(cls, name: str) -> AutoencoderFactoryCallable:
        """Return a registered autoencoder factory."""
        try:
            return cls._methods[name]
        except KeyError as error:
            raise ValueError(f"Unknown autoencoder method: {name}") from error

    @classmethod
    def model(cls, method: str, name: str) -> AutoencoderModelDescriptor:
        """Return a named model descriptor for a method."""
        try:
            return cls._models[method][name]
        except KeyError as error:
            raise ValueError(f"Unknown autoencoder model: {method}/{name}") from error

    @classmethod
    def methods(cls) -> tuple[str, ...]:
        """Return registered method names in deterministic order."""
        return tuple(sorted(cls._methods))

    @classmethod
    def models(cls, method: str) -> tuple[str, ...]:
        """Return named model variants registered for a method."""
        return tuple(sorted(cls._models.get(method, {})))
