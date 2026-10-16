"""Registered execution strategies, isolated workers and server script templates."""

import subprocess
import sys

import pytest

from pep_compass.orchestrator.backends.base import ExecutionBackend
from pep_compass.orchestrator.backends.subprocess import worker_command
from pep_compass.orchestrator.planning import plan_configuration
from pep_compass.orchestrator.runner import Orchestrator, RunExecution
from pep_compass.registry.execution.backends import backend_registry


def test_custom_backend_requires_no_orchestrator_dispatch_change(configuration):
    calls = []

    @backend_registry.register("fixture_custom")
    class CustomBackend(ExecutionBackend):
        def __init__(self, label="custom"):
            self.label = label

        def run(self, orchestrator, plan, plan_path):
            calls.append((self.label, plan_path))
            return tuple(
                RunExecution(entry.index, entry.job_id, "completed", entry.path)
                for entry in plan.entries
            )

    configuration.backend = {
        "strategy": "fixture_custom",
        "parameters": {"label": "server"},
    }
    record = Orchestrator().run(plan_configuration(configuration))[0]
    assert record.status == "completed"
    assert calls[0][0] == "server"


def test_worker_command_is_an_argument_vector_with_a_canonical_module(tmp_path):
    path = tmp_path / "plan with spaces; shell.json"
    command = worker_command(path, 3, skip_completed=True)
    assert command[:3] == [sys.executable, "-m", "pep_compass.orchestrator.cli"]
    assert str(path) in command and command[-1] == "--skip-completed"


def test_real_isolated_workers_read_the_same_frozen_plan(configuration):
    configuration.backend = {"strategy": "subprocess", "parameters": {"max_workers": 2}}
    configuration.input.mode = "individual"
    records = Orchestrator().run(plan_configuration(configuration))
    assert [record.index for record in records] == [0, 1, 2]
    assert all(record.status == "completed" for record in records), [
        record.error for record in records
    ]
    configuration.orchestrator.skip_completed = True
    assert all(
        record.status == "skipped"
        for record in Orchestrator().run(plan_configuration(configuration))
    )


def test_slurm_site_template_is_frozen_and_shell_syntax_is_valid(
    configuration, tmp_path
):
    template = tmp_path / "server.sh"
    template.write_text(
        """#!/bin/bash
#SBATCH --array=0-${array_max}
${directives}
set -euo pipefail
export SITE_NAME=fixture
cd ${working_directory}
${command}
""",
        encoding="utf-8",
    )
    configuration.backend = {
        "strategy": "slurm",
        "parameters": {
            "template_path": str(template),
            "settings": {"partition": "cpu", "cpus_per_task": 2},
        },
    }
    plan = plan_configuration(configuration)
    template.unlink()
    script = Orchestrator().run(plan)
    text = script.read_text()
    assert "SITE_NAME=fixture" in text and "--cpus-per-task=2" in text
    assert '"${SLURM_ARRAY_TASK_ID:?}"' in text
    assert "pep_compass.orchestrator.cli" in text
    subprocess.run(["bash", "-n", str(script)], check=True)


@pytest.mark.parametrize(
    "parameters",
    [
        {"settings": {"partition": "x\nmalicious"}},
        {"settings": {"cpus_per_task": True}},
        {"settings": {"unsupported": "x"}},
        {"template": "#!/bin/bash\necho no-worker\n"},
    ],
)
def test_slurm_rejects_invalid_directives_or_missing_worker(configuration, parameters):
    configuration.backend = {"strategy": "slurm", "parameters": parameters}
    with pytest.raises(ValueError):
        Orchestrator().run(plan_configuration(configuration))
