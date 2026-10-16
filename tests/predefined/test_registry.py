"""Named preset registration uses the same construction contract as components."""

import pytest

from pep_compass import Pipeline
from pep_compass.predefined import create, list_predefined, register


def test_builtin_predefined_factories_return_independent_editable_declarations():
    assert {"lebo", "simple"} <= set(list_predefined())
    first = create("lebo", mutation_candidates=2, oracle_calls=3)
    second = Pipeline.from_predefined("lebo", mutation_candidates=2, oracle_calls=3)
    assert first.to_schema().steps[0].node_id != second.to_schema().steps[0].node_id
    assert first.runner_settings.execution.oracle_calls == 3


def test_custom_predefined_decorator_needs_no_wrapper_changes():
    @register("fixture_empty")
    def custom(*, label="test"):
        """Return an editable declaration with application-owned settings."""
        return Pipeline()

    assert "fixture_empty" in list_predefined()
    assert isinstance(Pipeline.from_predefined("fixture_empty"), Pipeline)
    with pytest.raises(ValueError, match="Unknown predefined pipeline parameters"):
        create("fixture_empty", unknown=True)
