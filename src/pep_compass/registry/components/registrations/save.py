"""Decorated built-in saving component factories."""

from pep_compass.optimization.components.save.snapshot import Save
from pep_compass.registry.components.families.save import save_registry


@save_registry.register("snapshot", display_parameters=("name",))
def build_snapshot(*, name: str = "snapshot") -> Save:
    """Save an explicitly named candidate snapshot through the current backend."""
    if not isinstance(name, str) or not name:
        raise ValueError("Snapshot name must be a non-empty string.")
    return Save(name)
