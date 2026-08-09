"""Tests for typed YAML runtime configuration loading."""

from pathlib import Path

from pep_compass.runtime.configuration import load_runtime_configuration
import pytest


def test_minimal_yaml_loads_into_typed_configuration() -> None:
    """A valid YAML document must produce explicit configuration sections."""
    path = Path("tests/mock_data/configurations/minimal.yaml")

    configuration = load_runtime_configuration(path)

    assert configuration.experiment.name == "mock"
    assert configuration.autoencoder.method == "hydramp"
    assert configuration.autoencoder.model == "article_25"
    assert configuration.execution.backend == "local"


def test_loading_rejects_unknown_named_autoencoder_model(tmp_path) -> None:
    """Configuration validation must fail without loading model weights."""
    path = tmp_path / "invalid.yaml"
    path.write_text(
        """
experiment:
  name: invalid
  input:
    sequences: [AA]
autoencoder:
  method: hydramp
  model: missing
  parameters:
    jacobian_eps: 0.001
    field_eps: 0.001
pipeline:
  steps: []
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown autoencoder model"):
        load_runtime_configuration(path)
