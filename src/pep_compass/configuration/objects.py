"""Portable constructor snapshots and direct import-based object construction."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from math import isfinite
from pathlib import PurePath
from types import MappingProxyType


def target_path(value):
    """Return the importable defining path of an object or class."""
    kind = value if isinstance(value, type) else type(value)
    if "<locals>" in kind.__qualname__:
        raise ValueError("Persisted components must be defined in importable modules.")
    return kind.__module__ + "." + kind.__qualname__


def locate(path):
    """Import one explicitly selected target, without a discovery catalog."""
    if not isinstance(path, str) or "." not in path:
        raise ValueError("A target must be a qualified import path.")
    module, _, name = path.rpartition(".")
    return getattr(import_module(module), name)


def plain(value):
    """Snapshot constructor values without retaining live computational state."""
    if isinstance(value, Configurable):
        return value.to_config()
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Configuration mappings require string keys.")
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, PurePath):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and isfinite(value):
        return value
    if hasattr(value, "detach"):
        return plain(value.detach().cpu().tolist())
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    raise TypeError(f"Non-portable constructor parameter: {type(value).__name__}.")


def freeze(value):
    """Return recursively immutable configuration containers."""
    value = plain(value)
    if isinstance(value, dict):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(freeze(item) for item in value)
    return value


def construct(value):
    """Instantiate explicit constructor snapshots, recursively and without weights.

    :param value: Plain values or mappings containing a qualified ``_target_``.
    :return: Fresh constructor objects; no global registrations are consulted.
    """
    if isinstance(value, Mapping):
        if "_target_" in value:
            target = locate(value["_target_"])
            parameters = {
                key: construct(item) for key, item in value.items() if key != "_target_"
            }
            return target(**parameters)
        return {key: construct(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [construct(item) for item in value]
    return value


class Configurable:
    """Expose one constructor as the source of configuration and defaults.

    Concrete constructors call ``_configure`` with their public arguments.
    This base owns copying, serialization and nested resource initialization;
    it does not select implementations or maintain a catalog.
    """

    def __init__(self):
        self._configure({})

    def _configure(self, parameters):
        self._configuration = freeze(parameters)

    @property
    def configuration(self):
        """Return an independent plain snapshot of all resolved constructor values."""
        return plain(getattr(self, "_configuration", {}))

    def to_config(self):
        """Return a Hydra-compatible direct constructor declaration."""
        return {"_target_": target_path(self), **self.configuration}

    def initialize(self, autoencoder=None):
        """Prepare declared nested resources only when compiling a pipeline.

        :param autoencoder: Shared initialized autoencoder.
        :return: This prepared object.
        """
        if hasattr(self, "autoencoder") and getattr(self, "autoencoder") is None:
            self.autoencoder = autoencoder
        for value in vars(self).values():
            if isinstance(value, Configurable):
                value.initialize(autoencoder)
            elif isinstance(value, (list, tuple, dict)):
                for child in value.values() if isinstance(value, dict) else value:
                    if isinstance(child, Configurable):
                        child.initialize(autoencoder)
        prepare = getattr(self, "_initialize", None)
        if prepare is not None:
            prepare(autoencoder)
        return self
