"""Explicit loading of built-in pipeline declarations without model weights."""

from importlib import import_module


def load_predefined():
    """Import decorated preset factories; repeated imports are idempotent."""
    for name in ("lebo", "simple"):
        import_module(f"pep_compass.predefined.{name}")
