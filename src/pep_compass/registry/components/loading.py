"""Explicit loading of selected built-in families without creating models."""

from importlib import import_module


def load_components(*families: str) -> None:
    """Load selected families, or all built-ins when none are supplied.

    :param families: Public family names including ``autoencoder`` and ``save``.
    :raises ValueError: If a family is unknown.
    """
    modules = {
        "autoencoder": ("autoencoders",),
        "walker": ("sorbes", "walkers"),
        "mutation_generator": ("mutation_generators",),
        "filter": ("filters",),
        "oracle": ("oracles",),
        "save": ("save",),
    }
    for family in families or tuple(modules):
        if family not in modules:
            raise ValueError(f"Unknown built-in family: {family!r}.")
        for module in modules[family]:
            import_module("pep_compass.registry.components.registrations." + module)
