"""Sequential execution through the public Python pipeline wrapper."""

from pep_compass.orchestrator.backends.base import ExecutionBackend
from pep_compass.registry.execution.backends import register


@register("local")
class LocalBackend(ExecutionBackend):
    """Execute jobs in the current process, stopping according to failure policy."""

    def run(self, orchestrator, plan, plan_path):
        records = []
        for entry in plan.entries:
            record = orchestrator.run_entry(plan, entry)
            records.append(record)
            if record.status == "failed" and not plan.options.continue_on_error:
                break
        return tuple(records)
