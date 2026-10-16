"""Construct public wrapper objects from registered preset factories."""

from pep_compass.registry.pipelines.catalog import pipeline_registry
from pep_compass.registry.pipelines.loading import load_predefined


def create(name, **parameters):
    """Construct a fresh editable pipeline from a registered factory.

    :param name: Registered preset name.
    :param parameters: Public factory overrides.
    :return: An editable Pipeline; no model loading is required.
    :raises TypeError: If the factory violates the pipeline contract.
    """
    from pep_compass.core.wrapper.pipeline import Pipeline

    load_predefined()
    pipeline = pipeline_registry.build(name, parameters)
    if not isinstance(pipeline, Pipeline):
        raise TypeError("A predefined factory must return a Pipeline.")
    return pipeline
