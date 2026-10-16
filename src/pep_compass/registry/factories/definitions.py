"""Immutable registration metadata for named factories."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RegistryEntry(Generic[T]):
    """Describe one named factory owned by a :class:`Registry`.

    :param name: Stable configuration name.
    :param factory: Callable constructing the registered implementation.
    :param service_names: Composition-root services accepted by the factory.
    :param accepted_parameters: Explicit public parameter names. ``None``
        derives names from the factory signature.
    :param required_parameters: Explicit required public parameter names.
    """

    name: str
    factory: Callable[..., T]
    service_names: frozenset[str]
    accepted_parameters: frozenset[str] | None = None
    required_parameters: frozenset[str] | None = None
    config_type: type | None = None
    validator: Callable[[Mapping[str, Any]], None] | None = None
    normalizer: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None
    display_parameters: tuple[str, ...] = ()
    defaults: Mapping[str, Any] | None = None
    input_fields: Any = ()
    output_fields: Any = ()
    invalidated_prefixes: tuple[str, ...] = ()
    output_bound: Callable | None = None
    nested: Callable | Mapping | None = None
    optional_inputs: Any = ()
    model_provider: str | None = None

    def __post_init__(self):
        """Do not expose mutable copies of registered parameter defaults."""
        if self.defaults is not None:
            from pep_compass.data_schemas.graph import freeze

            object.__setattr__(self, "defaults", freeze(self.defaults))
