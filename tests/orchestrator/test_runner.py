"""Real wrapper execution, SQLite data, skipping and exclusive output ownership."""

import json
from pathlib import Path

from pep_compass.optimization.saving.reader import AnalysisReader
from pep_compass.orchestrator.planning import plan_configuration
from pep_compass.orchestrator.runner import Orchestrator


def test_local_execution_writes_canonical_bundle_and_readable_archive(configuration):
    plan = plan_configuration(configuration)
    records = Orchestrator().run(plan)
    assert [record.status for record in records] == ["completed"]
    entry = plan.entries[0]
    assert (Path(entry.path) / "settings.json").is_file()
    assert (Path(entry.path) / "results.sqlite").is_file()
    reader = AnalysisReader(entry.path)
    try:
        assert reader.summary["status"] == "completed"
        assert len(reader.get_archive()) == 2
        assert set(reader.get_archive().sequence) <= {"AA", "AC", "AD"}
    finally:
        reader.close()
    status = json.loads(
        (
            Path(plan.output_directory) / "statuses" / (entry.job_id + ".json")
        ).read_text()
    )
    assert status["run_id"] == records[0].run_id


def test_repeated_run_requires_explicit_skip_and_preserves_result(configuration):
    first = Orchestrator().run(plan_configuration(configuration))[0]
    repeated = Orchestrator().run(plan_configuration(configuration))[0]
    assert repeated.status == "failed" and "Output already exists" in repeated.error
    configuration.orchestrator.skip_completed = True
    skipped = Orchestrator().run(plan_configuration(configuration))[0]
    assert skipped.status == "skipped" and skipped.run_id == first.run_id


def test_skip_rejects_changed_settings(configuration):
    record = Orchestrator().run(plan_configuration(configuration))[0]
    settings = Path(record.path) / "settings.json"
    original = settings.read_bytes()
    # Change a real manifest field, rather than presentation whitespace.
    value = json.loads(original)
    value["inputs"] = ["GG"]
    settings.write_text(json.dumps(value))
    configuration.orchestrator.skip_completed = True
    failed = Orchestrator().run(plan_configuration(configuration))[0]
    assert failed.status == "failed" and "mismatch" in failed.error


def test_existing_job_lock_is_never_removed_or_overwritten(configuration):
    plan = plan_configuration(configuration)
    entry = plan.entries[0]
    lock = Path(plan.output_directory) / ".locks" / (entry.job_id + ".lock")
    lock.parent.mkdir(parents=True)
    lock.write_text("other-worker")
    record = Orchestrator().run_entry(plan, entry)
    assert record.status == "failed"
    assert lock.read_text() == "other-worker"
    assert not (
        Path(plan.output_directory) / "statuses" / (entry.job_id + ".json")
    ).exists()


def test_memory_backend_returns_summary_without_durable_result(configuration):
    configuration.runner.storage.backend = "memory"
    record = Orchestrator().run(plan_configuration(configuration))[0]
    assert record.status == "completed"
    assert record.path is None
    assert record.summary["status"] == "completed"
