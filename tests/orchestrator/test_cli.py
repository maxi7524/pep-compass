"""Actual Hydra CLI overrides and multirun with registered deterministic resources."""

import json
import subprocess
import sys

CLI = [sys.executable, "-m", "pep_compass.orchestrator.cli"]


def test_cli_plan_is_model_free_and_list_uses_registries(tmp_path):
    process = subprocess.run(
        [
            *CLI,
            "plan",
            "--plugin",
            "tests.fixtures.orchestration",
            "pipeline=fixture",
            "++runner.execution.device=cuda:99",
            f"hydra.run.dir={tmp_path}/hydra",
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    assert '"device": "cuda:99"' in process.stdout
    listed = subprocess.run(
        [*CLI, "list", "--plugin", "tests.fixtures.orchestration"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "fixture" in json.loads(listed.stdout)["pipelines"]


def test_cli_multirun_uses_hydra_seed_sweep_and_shared_wrapper(tmp_path):
    process = subprocess.run(
        [
            *CLI,
            "run",
            "--plugin",
            "tests.fixtures.orchestration",
            "--multirun",
            "pipeline=fixture",
            "++runner.execution.seed=3,4",
            "input.sequences=[AA,AC]",
            f"orchestrator.output_dir={tmp_path}/results",
            f"hydra.sweep.dir={tmp_path}/hydra",
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr + process.stdout[-2000:]
    manifests = sorted((tmp_path / "results").glob("*/runs/*/settings.json"))
    assert len(manifests) == 2
    assert {
        json.loads(path.read_text())["runner_settings"]["execution"]["seed"]
        for path in manifests
    } == {3, 4}
    assert all((path.parent / "results.sqlite").is_file() for path in manifests)


def test_cli_reports_worker_index_errors(tmp_path):
    process = subprocess.run(
        [*CLI, "worker", str(tmp_path / "missing.json"), "--index", "-1"],
        capture_output=True,
        text=True,
    )
    assert process.returncode != 0
