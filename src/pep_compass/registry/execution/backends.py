"""Decorator registration and explicit loading of execution backends."""

from importlib import import_module

from pep_compass.orchestrator.backends.base import ExecutionBackend
from pep_compass.registry.factories.registry import Registry

backend_registry = Registry("execution backend", expected_type=ExecutionBackend)
register = backend_registry.register


def load_backends():
    """Load built-in strategies; imported plugins may register additional ones."""
    for name in ("local", "subprocess", "slurm"):
        import_module(f"pep_compass.orchestrator.backends.{name}")
