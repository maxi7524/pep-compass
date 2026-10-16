"""Named defaults for assembling pipelines; strategies own their parameters."""


def default_autoencoder():
    """Return a fresh model selection; numerical defaults belong to its registry."""
    from pep_compass.data_schemas.graph import AutoencoderSpec

    return AutoencoderSpec("hydramp", "article_25")
