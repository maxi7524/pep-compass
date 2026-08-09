"""Tests for versioned runtime-result discovery."""

from pathlib import Path

from pep_compass.analysis.reader import ExperimentReader
from pep_compass.runtime.configuration.schema import (
    AutoencoderConfiguration,
    ExperimentConfiguration,
    RuntimeConfiguration,
)
from pep_compass.runtime.output import ResultWriter
from pep_compass.runtime.planning.plan import materialize_execution_plan
from pep_compass.runtime.runner import RuntimeRunner
from tests.fixtures.workflows import MockWorkflow


def test_reader_exposes_runtime_results_as_logical_dataset(tmp_path) -> None:
    """Reader discovery must produce both selections and ExperimentDataset."""
    configuration = RuntimeConfiguration(
        experiment=ExperimentConfiguration(
            name="reader",
            input={"sequences": ["AA"]},
        ),
        autoencoder=AutoencoderConfiguration("mock", "default"),
        pipeline={},
    )
    plan = materialize_execution_plan(configuration, working_directory=Path.cwd())
    RuntimeRunner(configuration, MockWorkflow(), ResultWriter(tmp_path)).run(plan)

    reader = ExperimentReader(tmp_path)

    assert len(reader.runs) == 1
    assert reader.dataset.schema_version == "1"
    assert reader.dataset.runs[0].identity.run_id == "run_00000"
    assert set(reader.dataset.tables) == {"candidates", "steps"}
    assert reader.dataset.tables["steps"].count_rows() == 2
    steps = reader.select().collect("steps")
    assert steps["step_name"].tolist() == ["SuffixStep", "Flow"]
