"""One shared deterministic configuration for orchestration tests."""

import pytest

from pep_compass.orchestrator.configuration import compose_configuration


@pytest.fixture
def configuration(tmp_path):
    """Use the registered fixture preset and isolated durable output."""
    return compose_configuration(
        [
            "pipeline=fixture",
            "input.sequences=[AA,AC,AD]",
            f"orchestrator.output_dir={tmp_path}",
        ],
        plugins=["tests.fixtures.orchestration"],
    )
