"""Frozen identity, input modes, portable serialization and model-free validation."""

import json

import pytest
from omegaconf import OmegaConf

from pep_compass.orchestrator.planning import load_plan, plan_configuration, save_plan


def test_repeated_plans_reuse_persisted_node_ids(configuration):
    first, path = save_plan(plan_configuration(configuration))
    second, _ = save_plan(plan_configuration(configuration))
    assert first.schema.to_dict() == second.schema.to_dict()
    assert first.entries == second.entries
    assert load_plan(path).schema.to_dict() == first.schema.to_dict()


def test_dispatch_policy_does_not_change_computation_identity(configuration):
    first = plan_configuration(configuration)
    configuration.orchestrator.skip_completed = True
    configuration.backend = {"strategy": "subprocess", "parameters": {"max_workers": 2}}
    second = plan_configuration(configuration)
    assert first.fingerprint == second.fingerprint
    assert [entry.job_id for entry in first.entries] == [
        entry.job_id for entry in second.entries
    ]


def test_input_changes_and_seed_changes_have_new_job_identity(configuration):
    first = plan_configuration(configuration)
    OmegaConf.update(configuration, "runner.execution.seed", 21, force_add=True)
    assert first.fingerprint != plan_configuration(configuration).fingerprint
    configuration.input.sequences = ["GG"]
    assert first.fingerprint != plan_configuration(configuration).fingerprint


@pytest.mark.parametrize(
    "extension,content",
    [
        ("csv", "sequence\nAA\nAC\n"),
        ("fa", ">first\nA\nA\n>second\nAC\n"),
        ("txt", "AA\nAC\n"),
    ],
)
def test_relative_input_files_and_individual_jobs(
    configuration, tmp_path, extension, content
):
    source = tmp_path / ("input." + extension)
    source.write_text(content, encoding="utf-8")
    configuration.input.sequences = None
    configuration.input.path = source.name
    configuration.input.mode = "individual"
    plan = plan_configuration(configuration, working_directory=tmp_path)
    assert [entry.sequences for entry in plan.entries] == [("AA",), ("AC",)]
    assert len({entry.path for entry in plan.entries}) == 2


@pytest.mark.parametrize(
    "patch",
    [
        {"limit": 0},
        {"mode": "unknown"},
        {"sequences": "AA"},
        {"sequences": []},
        {"path": "missing.txt"},
    ],
)
def test_invalid_inputs_are_rejected(configuration, patch):
    values = OmegaConf.to_container(configuration, resolve=True)
    values["input"].update(patch)
    with pytest.raises(ValueError):
        plan_configuration(values)


def test_frozen_plan_detects_corruption(configuration):
    plan, path = save_plan(plan_configuration(configuration))
    envelope = json.loads(path.read_text())
    envelope["payload"]["entries"][0]["sequences"] = ["tampered"]
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="integrity"):
        load_plan(path)


def test_orchestrator_owns_job_output_paths(configuration):
    configuration.runner.storage.path = "shared-output"
    with pytest.raises(ValueError, match="assigns storage.path"):
        plan_configuration(configuration)


def test_launch_changes_are_frozen_without_changing_computation_node_ids(configuration):
    first, first_path = save_plan(plan_configuration(configuration))
    configuration.orchestrator.skip_completed = True
    configuration.backend = {"strategy": "slurm", "parameters": {}}
    second, launch_path = save_plan(plan_configuration(configuration))
    assert first_path != launch_path
    assert first.entries == second.entries
    assert first.schema.to_dict() == second.schema.to_dict()
    restored = load_plan(launch_path)
    assert restored.options.skip_completed is True
    assert restored.backend["strategy"] == "slurm"
    assert load_plan(first_path).backend["strategy"] == "local"


def test_frozen_plugin_source_is_checked(configuration, monkeypatch):
    plan, path = save_plan(plan_configuration(configuration))
    monkeypatch.setattr(
        "pep_compass.orchestrator.planning.plugin_fingerprints",
        lambda modules: {"modified": "digest"},
    )
    with pytest.raises(ValueError, match="plugin source"):
        load_plan(path)
