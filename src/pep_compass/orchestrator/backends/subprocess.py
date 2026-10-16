"""Bounded independent worker processes consuming immutable plans."""

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from pep_compass.orchestrator.backends.base import ExecutionBackend
from pep_compass.utils.logger import get_custom_logger
from pep_compass.utils.validation import validate_integer

logger = get_custom_logger(__name__)


def worker_command(plan_path, index, *, skip_completed=False):
    """Return an argument vector, with no shell interpolation.

    :param plan_path: Frozen plan JSON.
    :param index: Zero-based job index.
    :param skip_completed: Explicit skip policy forwarded to the isolated worker.
    :return: Python executable and canonical CLI arguments.
    """
    command = [
        sys.executable,
        "-m",
        "pep_compass.orchestrator.cli",
        "worker",
        str(plan_path),
        "--index",
        str(index),
    ]
    if skip_completed:
        command.append("--skip-completed")
    return command


def execute_worker(plan, entry, plan_path):
    """Run one isolated worker and validate its structured terminal record."""
    from pep_compass.orchestrator.runner import RunExecution

    logger.info("Starting worker job=%s.", entry.job_id)
    process = subprocess.run(
        worker_command(
            plan_path, entry.index, skip_completed=plan.options.skip_completed
        ),
        cwd=plan.working_directory,
        capture_output=True,
        text=True,
    )
    messages = [
        line.removeprefix("PEP_COMPASS_RESULT=")
        for line in process.stdout.splitlines()
        if line.startswith("PEP_COMPASS_RESULT=")
    ]
    if messages:
        record = RunExecution(**json.loads(messages[-1]))
        if (
            record.index != entry.index
            or record.job_id != entry.job_id
            or (process.returncode == 0) != (record.status != "failed")
        ):
            raise RuntimeError(
                "Worker exit status or job identity does not match its result."
            )
        return record
    return RunExecution(
        entry.index,
        entry.job_id,
        "failed",
        entry.path,
        error=f"Worker exited {process.returncode}: {process.stderr[-2000:]}",
    )


class SubprocessBackend(ExecutionBackend):
    """Dispatch independent jobs in bounded isolated Python processes.

    :param max_workers: Maximum simultaneously running workers; default 1.
    """

    def __init__(self, max_workers=1):
        validate_integer(max_workers, "backend.max_workers", 1)
        self.max_workers = max_workers

    def run(self, orchestrator, plan, plan_path):
        if orchestrator.autoencoder is not None:
            raise ValueError(
                "Isolated workers require serializable resource declarations with direct import targets."
            )
        records = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(execute_worker, plan, entry, plan_path): entry
                for entry in plan.entries
            }
            failed = False
            for future in as_completed(futures):
                if future.cancelled():
                    continue
                record = future.result()
                records.append(record)
                if (
                    record.status == "failed"
                    and not plan.options.continue_on_error
                    and not failed
                ):
                    failed = True
                    for pending in futures:
                        pending.cancel()
        return tuple(sorted(records, key=lambda record: record.index))
