"""Named pipeline factories shared by Python discovery and Hydra groups."""

from pep_compass.registry.factories.registry import Registry

pipeline_registry = Registry("predefined pipeline")
register = pipeline_registry.register


def list_predefined():
    """List built-in and explicitly imported application presets."""
    from pep_compass.registry.pipelines.loading import load_predefined

    load_predefined()
    return pipeline_registry.names()
