"""Typed, validated registries for factories selected from configuration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict, fields, is_dataclass
from inspect import Parameter, isclass, signature
from typing import Any, Generic, TypeVar

from pep_compass.registry.factories.definitions import RegistryEntry
from pep_compass.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)

T = TypeVar("T")


class Registry(Generic[T]):
    """Register, validate, and build one namespaced family of factories.

    Registration is strict: a duplicate name is always an error. This avoids
    import-order-dependent implementation replacement.

    :param namespace: Human-readable namespace included in diagnostics.
    :param expected_type: Optional runtime type required from each factory.
    """

    def __init__(self, namespace: str, *, expected_type: type[T] | None = None) -> None:
        if not namespace:
            raise ValueError("Registry namespace cannot be empty.")
        self.namespace = namespace
        self.expected_type = expected_type
        self._entries: dict[str, RegistryEntry[T]] = {}

    @property
    def entries(self) -> Mapping[str, Callable[..., T]]:
        """Return a read-only-compatible view of registered factories."""
        return {name: entry.factory for name, entry in self._entries.items()}

    def register(
        self,
        name: str,
        *,
        services: set[str] | frozenset[str] | None = None,
        accepted_parameters: set[str] | frozenset[str] | None = None,
        required_parameters: set[str] | frozenset[str] | None = None,
        config_type: type | None = None,
        validator: Callable[[Mapping[str, Any]], None] | None = None,
        normalizer: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
        display_parameters: tuple[str, ...] = (),
        defaults: Mapping[str, Any] | None = None,
        input_fields=(),
        output_fields=(),
        invalidated_prefixes=(),
        output_bound=None,
        nested=None,
        optional_inputs=(),
        model_provider=None,
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Return a decorator registering one factory.

        :param name: Stable configuration name.
        :param services: Names injected only by the composition root.
        :param accepted_parameters: Explicit YAML parameter allowlist.
        :param required_parameters: Explicit required YAML parameters.
        :return: Decorator preserving the original callable.
        :raises ValueError: If the name is invalid or already registered.
        """
        normalized = self._normalize_name(name)
        if normalized in self._entries:
            raise ValueError(
                f"Duplicate registration in {self.namespace}: {normalized!r}."
            )

        def decorator(factory: Callable[..., T]) -> Callable[..., T]:
            if normalized in self._entries:
                raise ValueError(
                    f"Duplicate registration in {self.namespace}: {normalized!r}."
                )
            if not callable(factory):
                raise TypeError("Registered implementations must be callable.")
            if isclass(factory) and self.expected_type is not None:
                if not issubclass(factory, self.expected_type):
                    raise TypeError(
                        f"{self.namespace} implementations must inherit from "
                        f"{self.expected_type.__name__}."
                    )
            self._entries[normalized] = RegistryEntry(
                name=normalized,
                factory=factory,
                service_names=frozenset(services or ()),
                accepted_parameters=(
                    frozenset(accepted_parameters)
                    if accepted_parameters is not None
                    else None
                ),
                required_parameters=(
                    frozenset(required_parameters)
                    if required_parameters is not None
                    else None
                ),
                config_type=config_type,
                validator=validator,
                normalizer=normalizer,
                display_parameters=tuple(display_parameters),
                defaults=defaults,
                input_fields=input_fields,
                output_fields=output_fields,
                invalidated_prefixes=tuple(invalidated_prefixes),
                output_bound=output_bound,
                nested=nested,
                optional_inputs=optional_inputs,
                model_provider=model_provider,
            )
            logger.debug(
                "Registered implementation namespace=%s name=%s.",
                self.namespace,
                normalized,
            )
            return factory

        return decorator

    def names(self) -> tuple[str, ...]:
        """Return registered names in deterministic order."""
        return tuple(sorted(self._entries))

    def factory(self, name: str) -> Callable[..., T]:
        """Return a registered factory.

        :raises ValueError: If ``name`` is unknown.
        """
        return self._entry(name).factory

    def definition(self, name):
        """Return the immutable registration record, including graph contracts."""
        return self._entry(name)

    def validate(self, name: str, parameters: Mapping[str, Any]) -> None:
        """Validate public parameters without constructing an implementation.

        :param name: Registered strategy name.
        :param parameters: YAML-provided strategy parameters.
        :raises ValueError: If parameters are unknown, missing, or override a
            composition-root service.
        """
        entry = self._entry(name)
        if not isinstance(parameters, Mapping) or any(
            not isinstance(key, str) for key in parameters
        ):
            raise ValueError(
                f"{self.namespace} parameters must be a string-keyed mapping."
            )
        supplied = set(parameters)
        forbidden = supplied & entry.service_names
        if forbidden:
            raise ValueError(
                f"{self.namespace} parameters cannot override services: "
                f"{sorted(forbidden)}."
            )
        accepted, required = self._parameter_contract(entry)
        unknown = supplied - accepted
        missing = required - supplied
        if unknown:
            raise ValueError(f"Unknown {self.namespace} parameters: {sorted(unknown)}.")
        if missing:
            raise ValueError(f"Missing {self.namespace} parameters: {sorted(missing)}.")
        from pep_compass.data_schemas.graph import thaw

        values = {**thaw(entry.defaults or {}), **thaw(parameters)}
        if entry.config_type is not None:
            from pep_compass.registry.factories.parameters import validate_config_types

            validate_config_types(entry.config_type, values)
            entry.config_type(**values)
        if entry.validator is not None:
            entry.validator(values)

    def configure(
        self, name: str, parameters: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Resolve public defaults and validate without constructing a component.

        :param name: Registered strategy name.
        :param parameters: Explicit overrides; caller-owned values are copied.
        :return: Independent resolved parameter mapping.
        :raises ValueError: If parameters violate the registered contract.
        """
        entry = self._entry(name)
        from pep_compass.data_schemas.graph import thaw

        if parameters is not None and not isinstance(parameters, Mapping):
            raise ValueError("Strategy parameters must be a mapping.")
        supplied = thaw(parameters or {})
        self.validate(name, supplied)
        if entry.config_type is not None:
            value = entry.config_type(**{**thaw(entry.defaults or {}), **supplied})
            values = asdict(value) if is_dataclass(value) else dict(value)
        else:
            accepted, _ = self._parameter_contract(entry)
            values = {
                key: deepcopy(parameter.default)
                for key, parameter in signature(entry.factory).parameters.items()
                if key in accepted and parameter.default is not Parameter.empty
            }
            values.update(thaw(entry.defaults or {}))
            values.update(supplied)
        if entry.normalizer is not None:
            values = dict(entry.normalizer(values))
        return thaw(values)

    def describe(self, name: str) -> dict[str, Any]:
        """Describe a strategy's parameters and dependencies without building it.

        :param name: Registered strategy name.
        :return: Structured documentation suitable for notebook presentation.
        """
        from inspect import getdoc

        from pep_compass.data_schemas.graph import thaw

        entry = self._entry(name)
        accepted, required = self._parameter_contract(entry)
        signature_parameters = signature(entry.factory).parameters
        config_fields = (
            {field.name: field for field in fields(entry.config_type)}
            if entry.config_type is not None
            else {}
        )
        from dataclasses import MISSING

        parameters = []
        for key in sorted(accepted):
            field = config_fields.get(key)
            parameter = signature_parameters.get(key)
            default = (entry.defaults or {}).get(key, Parameter.empty)
            annotation = parameter.annotation if parameter is not None else Any
            description = ""
            if field is not None:
                annotation = field.type
                if default is Parameter.empty:
                    default = (
                        field.default
                        if field.default is not MISSING
                        else (
                            field.default_factory()
                            if field.default_factory is not MISSING
                            else Parameter.empty
                        )
                    )
                description = field.metadata.get("description", "")
            elif default is Parameter.empty and parameter is not None:
                default = parameter.default
            type_name = (
                annotation.__name__
                if isinstance(annotation, type)
                else str(annotation).removeprefix("typing.")
            )
            parameters.append(
                {
                    "name": key,
                    "type": type_name,
                    "required": key in required,
                    "has_default": default is not Parameter.empty,
                    "default": None if default is Parameter.empty else thaw(default),
                    "description": description,
                }
            )
        nested = entry.nested() if callable(entry.nested) else entry.nested or {}
        models = {}
        if entry.model_provider is not None:
            from pep_compass.registry.models.catalog import model_catalog

            models = {
                "models": list(model_catalog.names(entry.model_provider)),
                "model_descriptions": model_catalog.describe(entry.model_provider),
            }
        return {
            "strategy": entry.name,
            "namespace": self.namespace,
            **models,
            "description": getdoc(entry.factory) or "",
            "parameters": parameters,
            "services": sorted(
                name for name in entry.service_names if name in signature_parameters
            ),
            "input_fields": "configuration-dependent"
            if callable(entry.input_fields)
            else list(entry.input_fields),
            "output_fields": "configuration-dependent"
            if callable(entry.output_fields)
            else list(entry.output_fields),
            "optional_inputs": "configuration-dependent"
            if callable(entry.optional_inputs)
            else list(entry.optional_inputs),
            "nested_strategies": {
                slot: [registry.describe(method) for method in registry.names()]
                for slot, registry in nested.items()
            },
            "display_parameters": list(entry.display_parameters),
        }

    def build(
        self,
        name: str,
        parameters: Mapping[str, Any] | None = None,
        *,
        services: Mapping[str, Any] | None = None,
    ) -> T:
        """Validate and construct one registered implementation.

        :param name: Registered strategy name.
        :param parameters: YAML-provided parameters.
        :param services: Composition-root-only dependencies.
        :return: Constructed implementation.
        :raises TypeError: If the factory returns an unexpected type.
        """
        resolved_parameters = self.configure(name, parameters)
        entry = self._entry(name)
        available_services = dict(services or {})
        factory_parameters = signature(entry.factory).parameters
        declared_services = {
            service for service in entry.service_names if service in factory_parameters
        }
        missing_services = {
            service
            for service in declared_services
            if service not in available_services
            and factory_parameters[service].default is Parameter.empty
        }
        if missing_services:
            raise ValueError(
                f"{self.namespace} strategy {name!r} requires services: "
                f"{sorted(missing_services)}."
            )
        arguments = {
            **resolved_parameters,
            **{
                service: available_services[service]
                for service in declared_services
                if service in available_services
            },
        }
        logger.debug(
            "Building implementation namespace=%s name=%s.", self.namespace, name
        )
        value = entry.factory(**arguments)
        if self.expected_type is not None and not isinstance(value, self.expected_type):
            raise TypeError(
                f"{self.namespace} strategy {name!r} returned "
                f"{type(value).__name__}, expected {self.expected_type.__name__}."
            )
        return value

    def resolve(
        self,
        target: str | type[T] | T,
        parameters: Mapping[str, Any] | None = None,
        *,
        services: Mapping[str, Any] | None = None,
    ) -> T:
        """Resolve a registered name, implementation class, or ready instance.

        :param target: Component name, class, or initialized instance.
        :param parameters: Constructor parameters; forbidden for ready instances.
        :param services: Dependencies injected by the composition root.
        :return: Validated component instance.
        :raises TypeError: If the target violates the family contract.
        :raises ValueError: If parameters or dependencies are invalid.
        """
        if isinstance(target, str):
            return self.build(target, parameters, services=services)
        if not isclass(target):
            if self.expected_type is None or not isinstance(target, self.expected_type):
                raise TypeError(f"Invalid {self.namespace} component instance.")
            if parameters:
                raise ValueError(
                    "Ready component instances cannot accept constructor parameters."
                )
            return target
        temporary = Registry(self.namespace, expected_type=self.expected_type)
        temporary.register(
            target.__name__,
            services=set(services or ()),
        )(target)
        return temporary.build(target.__name__, parameters, services=services)

    def _entry(self, name: str) -> RegistryEntry[T]:
        normalized = self._normalize_name(name)
        try:
            return self._entries[normalized]
        except KeyError as error:
            raise ValueError(
                f"Unknown {self.namespace} strategy: {normalized!r}. "
                f"Available: {list(self.names())}."
            ) from error

    @staticmethod
    def _normalize_name(name: str) -> str:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Registry name must be a non-empty string.")
        return name.strip()

    @staticmethod
    def _parameter_contract(entry: RegistryEntry[T]) -> tuple[set[str], set[str]]:
        source = entry.factory
        parameters = signature(source).parameters
        if entry.accepted_parameters is None:
            accepted = {
                name
                for name, parameter in parameters.items()
                if parameter.kind
                not in {Parameter.VAR_KEYWORD, Parameter.VAR_POSITIONAL}
                and name not in entry.service_names
                and name != "self"
            }
        else:
            accepted = set(entry.accepted_parameters)
        if entry.required_parameters is not None:
            required = set(entry.required_parameters)
        else:
            required = {
                name
                for name, parameter in parameters.items()
                if name not in entry.service_names
                and name != "self"
                and parameter.default is Parameter.empty
                and parameter.kind
                not in {Parameter.VAR_KEYWORD, Parameter.VAR_POSITIONAL}
            }
        if entry.config_type is not None:
            from dataclasses import MISSING

            declared = fields(entry.config_type)
            accepted = {field.name for field in declared}
            required = {
                field.name
                for field in declared
                if field.default is MISSING and field.default_factory is MISSING
            }
        required -= set(entry.defaults or {})
        return accepted, required
