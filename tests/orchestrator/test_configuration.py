"""Hydra group replacement, interpolation and wrapper declaration contracts."""

import pytest
from omegaconf import OmegaConf

from pep_compass.orchestrator.configuration import compose_configuration
from pep_compass.orchestrator.declarations import create_pipeline
from pep_compass.orchestrator.planning import plan_configuration


def test_groups_and_preset_defaults_are_model_free(configuration, monkeypatch):
    from pep_compass.autoencoder.factory import AutoencoderFactory

    monkeypatch.setattr(
        AutoencoderFactory,
        "build",
        lambda *args, **kwargs: pytest.fail("Planning constructed a model"),
    )
    plan = plan_configuration(configuration)
    assert plan.entries[0].settings.execution.seed == 7
    assert plan.entries[0].settings.execution.oracle_calls == 3
    assert plan.schema.autoencoder.strategy == "fixture_encoder"
    assert plan.schema.steps[0].parameters["count"] == 2


def test_explicit_null_overrides_preset_threshold(configuration):
    OmegaConf.update(
        configuration, "runner.execution.oracle_calls", None, force_add=True
    )
    plan = plan_configuration(configuration)
    assert plan.entries[0].settings.execution.oracle_calls is None


def test_hydra_overrides_groups_and_runner_settings(tmp_path):
    configuration = compose_configuration(
        [
            "pipeline=fixture",
            "backend=subprocess",
            "backend.parameters.max_workers=2",
            "++runner.execution.seed=19",
            "input.sequences=[AA]",
            f"orchestrator.output_dir={tmp_path}",
        ],
        plugins=["tests.fixtures.orchestration"],
    )
    plan = plan_configuration(configuration)
    assert plan.backend == {"strategy": "subprocess", "parameters": {"max_workers": 2}}
    assert plan.entries[0].settings.execution.seed == 19


def test_custom_config_interpolation_and_explicit_steps(tmp_path):
    (tmp_path / "custom.yaml").write_text(
        """pipeline:
  autoencoder: {strategy: fixture_encoder, model: default}
  steps:
    - {type: loop, iterations: 2, steps: [{component: oracle, strategy: hydrophobicity}]}
input: {sequences: [AA], mode: batch}
orchestrator: {name: local, output_dir: results}
runner: {execution: {seed: 13}}
backend: {strategy: local, parameters: {}}
plugins: [tests.fixtures.orchestration]
""",
        encoding="utf-8",
    )
    config = compose_configuration(config_dir=tmp_path, config_name="custom")
    config.orchestrator.output_dir = "${orchestrator.name}/out"
    plan = plan_configuration(config, working_directory=tmp_path)
    assert plan.schema.steps[0].iterations == 2
    assert plan.output_directory == str(tmp_path / "local/out/local")


@pytest.mark.parametrize(
    "settings",
    [
        {"predefined": "simple", "steps": []},
        {"predefined": "simple", "parameters": {"seed": 2}},
        {"steps": [], "parameters": {}},
        {"steps": [], "unknown": True},
    ],
)
def test_invalid_wrapper_configuration_is_rejected(settings):
    with pytest.raises(ValueError):
        create_pipeline(settings)


def test_schema_round_trip_uses_public_wrapper(configuration):
    from pep_compass import Pipeline

    pipeline = Pipeline.from_predefined("fixture")
    restored = create_pipeline({"schema": pipeline.to_schema("json")})
    assert restored.to_schema().to_dict() == pipeline.to_schema().to_dict()


def test_unknown_parameters_fail_before_execution(configuration):
    values = OmegaConf.to_container(configuration, resolve=True)
    values["pipeline"]["parameters"]["count_typo"] = 9
    with pytest.raises(ValueError, match="Unknown predefined pipeline parameters"):
        plan_configuration(values)
