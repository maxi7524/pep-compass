"""Public Python API without registration or eager model initialization."""

__all__ = ["Pipeline", "PipelineBuilder", "PipelineSchema"]


def __getattr__(name):
    """Load explicitly requested interfaces without initializing model weights."""
    if name == "Pipeline":
        from pep_compass.core.wrapper.pipeline import Pipeline

        return Pipeline
    if name == "PipelineBuilder":
        from pep_compass.core.builder.builder import PipelineBuilder

        return PipelineBuilder
    if name == "PipelineSchema":
        from pep_compass.data_schemas.graph import PipelineSchema

        return PipelineSchema
    raise AttributeError(name)
