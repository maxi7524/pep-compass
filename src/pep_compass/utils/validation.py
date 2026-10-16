"""Strict reusable validation for Python and configuration declarations."""

from collections.abc import Mapping
from math import isfinite
from numbers import Real
from typing import Any


def validate_keys(value: Mapping[str, Any], allowed: set[str], path: str) -> None:
    """Reject unknown declaration keys.

    :param value: Mapping to inspect.
    :param allowed: Permitted string keys.
    :param path: Diagnostic configuration path.
    :raises ValueError: If the mapping contains unknown keys.
    """
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {path} keys: {sorted(unknown, key=str)}.")


def validate_integer(
    value: Any, path: str, minimum: int = 0, *, optional=False
) -> None:
    """Require a bounded integer without coercing booleans or fractions.

    :param value: Value to validate.
    :param path: Diagnostic configuration path.
    :param minimum: Inclusive lower bound.
    :param optional: Whether null is accepted.
    :raises ValueError: If the value violates the integer contract.
    """
    if optional and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{path} must be an integer >= {minimum}.")


def validate_positive(value: Any, path: str) -> None:
    """Require a finite positive real value.

    :param value: Value to validate.
    :param path: Diagnostic configuration path.
    :raises ValueError: If the numeric constraint is violated.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{path} must be finite and positive.")


def validate_declaration(value: Any, path: str) -> tuple[str, Mapping[str, Any]]:
    """Validate one nested strategy declaration.

    :param value: Mapping containing method and optional parameters.
    :param path: Diagnostic configuration path.
    :return: Method name and parameter mapping.
    :raises ValueError: If the declaration structure is invalid.
    """
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping.")
    validate_keys(value, {"method", "parameters"}, path)
    method = value.get("method")
    parameters = value.get("parameters", {})
    if not isinstance(method, str) or not method.strip():
        raise ValueError(f"{path}.method must be a non-empty string.")
    if not isinstance(parameters, Mapping):
        raise ValueError(f"{path}.parameters must be a mapping.")
    return method, parameters
