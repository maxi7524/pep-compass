"""Tests for translation from runtime mappings to core specifications."""

from pep_compass.core.specification import LoopSpecification
from pep_compass.runtime.configuration.pipeline import parse_pipeline_specification


def test_pipeline_parser_builds_typed_nested_declarations() -> None:
    """Runtime syntax parsing must end at the neutral core specification."""
    specification = parse_pipeline_specification(
        {
            "steps": [
                {
                    "loop": {
                        "iterations": 2,
                        "steps": [],
                    }
                }
            ]
        }
    )

    assert isinstance(specification.root.steps[0], LoopSpecification)
    assert specification.root.steps[0].iterations == 2
