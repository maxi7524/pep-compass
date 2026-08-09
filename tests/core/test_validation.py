"""Tests for non-fatal computation-graph diagnostics."""

from pep_compass.core.specification import (
    ComponentSpecification,
    FlowSpecification,
    LoopSpecification,
    PipelineSpecification,
)
from pep_compass.core.validation import diagnose_pipeline_specification


def test_diagnostics_reject_mutation_feedback_into_walker_loop() -> None:
    """A generic walker-generator loop must expose trajectory branching."""
    specification = PipelineSpecification(
        FlowSpecification(
            (
                LoopSpecification(
                    2,
                    FlowSpecification(
                        (
                            ComponentSpecification("walker", "sorbes"),
                            ComponentSpecification("mutation_generator", "mutang"),
                        )
                    ),
                ),
            )
        )
    )

    diagnostics = diagnose_pipeline_specification(specification)

    assert {diagnostic.code for diagnostic in diagnostics} == {
        "TRAJECTORY_FEEDBACK",
        "UNBOUNDED_EXPANSION",
    }
