"""Parameter validation and factory service injection."""

from __future__ import annotations


def validate_config_types(config_type, values):
    """Check dataclass annotations without silently coercing user parameters.

    Numeric float fields accept integers, but integer fields reject booleans.
    Semantic constraints remain in the configuration or registered validator.
    """
    from collections.abc import Mapping, Sequence
    from types import UnionType
    from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

    annotations = get_type_hints(config_type)

    def accepts(value, annotation):
        origin, arguments = get_origin(annotation), get_args(annotation)
        if annotation in {Any, object}:
            return True
        if origin in {Union, UnionType}:
            return any(accepts(value, item) for item in arguments)
        if origin is Literal:
            return value in arguments and any(
                type(value) is type(item) for item in arguments
            )
        if annotation is float:
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if annotation is int:
            return isinstance(value, int) and not isinstance(value, bool)
        if origin in {dict, Mapping}:
            return isinstance(value, Mapping) and (
                not arguments
                or all(
                    accepts(key, arguments[0]) and accepts(item, arguments[1])
                    for key, item in value.items()
                )
            )
        if origin in {list, tuple, Sequence}:
            if not isinstance(value, (list, tuple)):
                return False
            return not arguments or all(accepts(item, arguments[0]) for item in value)
        return isinstance(value, annotation)

    for name, value in values.items():
        annotation = annotations.get(name, Any)
        if not accepts(value, annotation):
            raise ValueError(
                f"{config_type.__name__}.{name} must match {annotation}, got {type(value).__name__}."
            )
