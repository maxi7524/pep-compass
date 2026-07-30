"""Dependency injection helper for registered strategy factories."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from inspect import signature
from typing import Any


def build_with_services(
    factory: Callable[..., Any],
    parameters: Mapping[str, Any],
    services: Mapping[str, Any] | None,
) -> Any:
    """Invoke a factory with only the services it explicitly declares.

    :param factory: Registered strategy factory or class.
    :type factory: Callable[..., Any]
    :param parameters: User-provided strategy parameters.
    :type parameters: Mapping[str, Any]
    :param services: Developer-owned services supplied by the composition root.
    :type services: Mapping[str, Any] | None
    :return: Constructed strategy.
    :rtype: Any
    :raises ValueError: If a user parameter attempts to replace a core service.
    """
    arguments = dict(parameters)
    for name, service in (services or {}).items():
        if name not in signature(factory).parameters:
            continue
        if name in arguments:
            raise ValueError(f"Strategy parameter cannot override core service: {name}")
        arguments[name] = service
    return factory(**arguments)
