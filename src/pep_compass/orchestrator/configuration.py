"""Hydra configuration groups derived from registered public factories."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

from hydra import compose, initialize_config_dir, initialize_config_module
from hydra.core.config_store import ConfigStore

from pep_compass.registry.components.loading import load_components
from pep_compass.registry.pipelines.catalog import list_predefined, pipeline_registry


@dataclass(frozen=True)
class OrchestrationSettings:
    """Independent-job identity, output root and failure policy."""

    name: str = "experiment"
    output_dir: str = "results"
    skip_completed: bool = False
    continue_on_error: bool = False

    def __post_init__(self):
        if (
            not isinstance(self.name, str)
            or not self.name
            or Path(self.name).name != self.name
            or self.name in {".", ".."}
        ):
            raise ValueError(
                "orchestrator.name must be a single non-empty path component."
            )
        if not isinstance(self.output_dir, str) or not self.output_dir:
            raise ValueError("orchestrator.output_dir must be a non-empty path.")
        for name in ("skip_completed", "continue_on_error"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"orchestrator.{name} must be a boolean.")


def load_plugins(modules):
    """Import explicitly requested registration modules before composing configs.

    :param modules: Python module names supplied by the application owner.
    :raises ValueError: If names are not a sequence of non-empty strings.
    """
    if isinstance(modules, (str, bytes)) or any(
        not isinstance(name, str) or not name for name in modules
    ):
        raise ValueError("plugins must contain Python module names.")
    for name in modules:
        import_module(name)


def register_configurations():
    """Publish preset and backend defaults without initializing models."""
    from dataclasses import asdict

    from pep_compass.registry.execution.backends import backend_registry, load_backends

    load_components()
    load_backends()
    store = ConfigStore.instance()
    for name in list_predefined():
        parameters = {
            p["name"]: p["default"] if p["has_default"] else "???"
            for p in pipeline_registry.describe(name)["parameters"]
            if p["name"] not in {"seed", "device"}
        }
        store.store(
            group="pipeline",
            name=name,
            node={"predefined": name, "parameters": parameters},
            provider="pep_compass",
        )
    for name in backend_registry.names():
        store.store(
            group="backend",
            name=name,
            node={
                "strategy": name,
                "parameters": {
                    p["name"]: p["default"] if p["has_default"] else "???"
                    for p in backend_registry.describe(name)["parameters"]
                },
            },
            provider="pep_compass",
        )
    store.store(
        group="orchestrator",
        name="default",
        node=asdict(OrchestrationSettings()),
        provider="pep_compass",
    )
    # REMARK: A patch preserves preset defaults; explicit null overrides a preset limit.
    store.store(
        group="runner",
        name="default",
        node={
            "execution": {},
            "storage": {"backend": "sqlite", "path": None},
            "results": {},
        },
        provider="pep_compass",
    )


def compose_configuration(
    overrides=(), *, config_dir=None, config_name="config", plugins=()
):
    """Compose Hydra defaults, interpolation and overrides from Python.

    :param overrides: Hydra override expressions.
    :param config_dir: Optional custom configuration directory.
    :param config_name: Root YAML name without its extension.
    :param plugins: Modules registering additional presets or backends.
    :return: Composed OmegaConf DictConfig; no models are constructed.
    """
    load_plugins(plugins)
    register_configurations()
    context = (
        initialize_config_module(
            version_base="1.3", config_module="pep_compass.orchestrator.conf"
        )
        if config_dir is None
        else initialize_config_dir(
            version_base="1.3", config_dir=str(Path(config_dir).resolve())
        )
    )
    with context:
        configuration = compose(config_name=config_name, overrides=list(overrides))
    from omegaconf import open_dict

    with open_dict(configuration):
        configuration.plugins = list(
            dict.fromkeys([*configuration.get("plugins", []), *plugins])
        )
    return configuration


def plugin_fingerprints(modules):
    """Hash explicitly imported plugin module files for frozen-plan identity.

    :param modules: Importable registration modules.
    :return: Mapping from module name to its source/artifact SHA-256.
    :raises ValueError: If a plugin has no inspectable file.
    """
    import hashlib
    import inspect

    result = {}
    for name in modules:
        module = import_module(name)
        source = inspect.getsourcefile(module) or getattr(module, "__file__", None)
        if source is None:
            raise ValueError(f"Plugin {name!r} has no inspectable source file.")
        result[name] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    return result
