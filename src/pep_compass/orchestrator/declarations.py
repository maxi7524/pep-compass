"""Instantiate the same public Python constructors selected by Hydra targets."""

from hydra.utils import instantiate

from pep_compass.core.wrapper import Pipeline


def create_pipeline(configuration, *, autoencoder=None, execution=None):
    """Construct lightweight wrapper declarations; validation never loads weights.

    :param configuration: Hydra mapping with a public pipeline or preset _target_.
    :param autoencoder: Optional already initialized Python resource.
    :param execution: Optional runner initialization defaults.
    :return: Editable Pipeline using exactly the Python component API.
    """
    pipeline = instantiate(configuration, _convert_="all")
    if not isinstance(pipeline, Pipeline):
        raise TypeError("pipeline._target_ must construct a Pipeline.")
    if autoencoder is not None:
        pipeline.add_autoencoder(autoencoder)
    pipeline.set_runner_settings(execution=execution)
    pipeline.validate()
    return pipeline
