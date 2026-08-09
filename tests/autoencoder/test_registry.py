"""Tests for explicit autoencoder method and model registration."""

import pep_compass.autoencoder.strategies  # noqa: F401

from pep_compass.autoencoder.registry import AutoencoderRegistry


def test_hydramp_method_and_named_model_are_registered_separately() -> None:
    """Implementation selection and weight/model selection must be independent."""
    assert AutoencoderRegistry.methods() == ("hydramp",)
    assert AutoencoderRegistry.models("hydramp") == ("article_25",)
    assert callable(AutoencoderRegistry.method("hydramp"))
    assert AutoencoderRegistry.model("hydramp", "article_25").parameters == {
        "model_name": "article_25"
    }
