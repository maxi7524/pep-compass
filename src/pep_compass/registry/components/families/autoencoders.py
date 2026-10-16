"""Registry of autoencoder methods and named model variants."""

from __future__ import annotations

from collections.abc import Callable

from pep_compass.autoencoder.base import Autoencoder
from pep_compass.registry.factories.registry import Registry
from pep_compass.registry.models.catalog import model_catalog
from pep_compass.registry.models.definitions import (
    AutoencoderModelDescriptor,
    ModelDescriptor,
)

AutoencoderFactoryCallable = Callable[..., Autoencoder]


class AutoencoderRegistry:
    """Store autoencoder factories and named model descriptors."""

    _method_registry: Registry[Autoencoder] = Registry(
        "autoencoder", expected_type=Autoencoder
    )

    @classmethod
    def register_method(
        cls,
        name: str,
        **contract,
    ) -> Callable[[AutoencoderFactoryCallable], AutoencoderFactoryCallable]:
        """Return a decorator registering one autoencoder implementation."""
        contract.setdefault("model_provider", f"autoencoder.{name}")
        return cls._method_registry.register(name, **contract)

    @classmethod
    def register_model(
        cls,
        method: str,
        descriptor: AutoencoderModelDescriptor,
    ) -> None:
        """Register a named model variant for an autoencoder method."""
        model_catalog.register(
            ModelDescriptor(
                provider=f"autoencoder.{method}",
                name=descriptor.name,
                directory=descriptor.directory or descriptor.name,
                artifacts=descriptor.artifacts,
                parameters=dict(descriptor.parameters),
                download_script=descriptor.download_script,
            )
        )

    @classmethod
    def method(cls, name: str) -> AutoencoderFactoryCallable:
        """Return a registered autoencoder factory."""
        return cls._method_registry.factory(name)

    @classmethod
    def validate(cls, method: str, parameters) -> None:
        """Validate constructor parameters through the shared registry."""
        cls._method_registry.validate(method, parameters)

    @classmethod
    def build(cls, method: str, parameters) -> Autoencoder:
        """Build and type-check an implementation through the shared registry."""
        return cls._method_registry.build(method, parameters)

    @classmethod
    def model(cls, method: str, name: str) -> ModelDescriptor:
        """Return a named model descriptor for a method."""
        try:
            return model_catalog.descriptor(f"autoencoder.{method}", name)
        except ValueError as error:
            raise ValueError(
                f"Unknown autoencoder model: {method}/{name}. "
                f"Available: {list(cls.models(method))}."
            ) from error

    @classmethod
    def methods(cls) -> tuple[str, ...]:
        """Return registered method names in deterministic order."""
        return cls._method_registry.names()

    @classmethod
    def describe(cls, method):
        """Describe model-adapter parameters without initializing weights."""
        return cls._method_registry.describe(method)

    @classmethod
    def configure(cls, method, parameters=None):
        """Resolve public adapter defaults through the common registry."""
        return cls._method_registry.configure(method, parameters)

    @classmethod
    def models(cls, method: str) -> tuple[str, ...]:
        """Return named model variants registered for a method."""
        return model_catalog.names(f"autoencoder.{method}")
