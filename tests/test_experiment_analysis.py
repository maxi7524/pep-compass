import csv
import json
from pathlib import Path

from pep_compass.experiments.analysis import LocalityExperiment
from pep_compass.experiments.analysis.methods import (
    filter_score_summary,
    retention_summary,
)
from pep_compass.experiments.runner.tracking import should_start_tracking


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _experiment(tmp_path: Path) -> LocalityExperiment:
    root = tmp_path / "results" / "locality" / "example"
    tracking = root / "grid_0000" / "tracking" / "experiment"
    tasks = root / "tasks"
    tracking.mkdir(parents=True)
    tasks.mkdir()
    task = {
        "experiment_id": "experiment",
        "output_path": str(root / "grid_0000"),
        "config": {
            "optimizer": {"name": "lebo", "lebo": {"candidate_strategy": "tandem"}},
            "tracking": {"level": "all"},
        },
    }
    (tasks / "task_000000.json").write_text(json.dumps(task), encoding="utf-8")
    _write_csv(
        root / "grid_manifest.csv",
        [{"grid_id": "grid_0000", "output_path": str(root / "grid_0000"), "parameters": '{"filter.temperature": 2.0}'}],
    )
    _write_csv(
        root / "run_manifest.csv",
        [{"task_id": "task_000000", "grid_id": "grid_0000", "name": "p", "sequence": "AAA", "repetition": 0, "seed": 1, "device": "cpu", "output_path": str(root / "grid_0000"), "task_file": str(tasks / "task_000000.json")}],
    )
    _write_csv(
        tracking / "enumeration_steps.csv",
        [{"run_id": "task_000000", "iteration_id": 2, "trajectory_id": 0, "step_id": 1, "proposed_count": 100, "post_limit_count": 20, "post_method_filter_count": 10, "post_constraint_filter_count": 5}],
    )
    _write_csv(
        tracking / "candidates.csv",
        [
            {"run_id": "task_000000", "iteration_id": 2, "trajectory_id": 0, "step_id": 1, "method_score": 1.0, "passed_method_filter": True},
            {"run_id": "task_000000", "iteration_id": 2, "trajectory_id": 0, "step_id": 1, "method_score": 0.0, "passed_method_filter": False},
        ],
    )
    return LocalityExperiment.open(tmp_path / "results" / "locality")


def test_tracking_thresholds_use_or_semantics() -> None:
    config = {"start_iteration": 5, "start_sequence_length": 10}
    assert not should_start_tracking(config, 4, "SHORT")
    assert should_start_tracking(config, 5, "SHORT")
    assert should_start_tracking(config, 1, "LONGPEPTIDE")
    assert should_start_tracking({}, 0, "A")


def test_manifest_catalog_and_streaming_summaries(tmp_path: Path) -> None:
    selection = _experiment(tmp_path).select(iteration_min=2)

    retention = retention_summary(selection, ["method"])
    scores = filter_score_summary(selection, ["method"])

    assert retention.loc[0, "limit_retention"] == 0.2
    assert retention.loc[0, "constraint_retention"] == 0.5
    assert scores.loc[0, "acceptance_rate"] == 0.5
    assert scores.loc[0, "score_mean"] == 0.5
