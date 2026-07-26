import csv
from pathlib import Path

import pytest

from pep_compass.optimization.lebo.trajectory_tracking import (
    CandidateEventSpool,
    CandidateProvenance,
    EnumerationStep,
    EnumerationTrace,
    LeboCSVTracker,
)


def _tracker(output: Path, level: str) -> LeboCSVTracker:
    return LeboCSVTracker(
        output_directory=output,
        run_id="run",
        level=level,
        candidate_strategy="tandem",
        objective_name="apex",
        objective_direction="minimize",
        objective_description="APEX result",
        objective_parameters={},
        store_latents=False,
    )


def _trace() -> EnumerationTrace:
    accepted = CandidateProvenance(
        sequence="ACD",
        parent_sequence="AAA",
        trajectory_id=0,
        step_id=1,
        source_iteration_id=1,
        node_id="step_1_candidate_0",
        parent_id="step_1",
    )
    rejected = CandidateProvenance(
        sequence="ACE",
        parent_sequence="AAA",
        trajectory_id=0,
        step_id=1,
        source_iteration_id=1,
        node_id="step_1_candidate_1",
        parent_id="step_1",
        passed_method_filter=False,
        passed_constraint_filter=False,
    )
    all_candidates = CandidateEventSpool()
    all_candidates.append(accepted)
    all_candidates.append(rejected)
    return EnumerationTrace(
        generated_count=2,
        accepted_count=1,
        candidates={accepted.sequence: accepted},
        steps=[
            EnumerationStep(
                trajectory_id=0,
                step_id=1,
                node_id="step_1",
                parent_id="root",
                parent_sequence="AAA",
                next_sequence="AAC",
                proposed_count=20,
                post_limit_count=2,
                post_method_filter_count=1,
                post_constraint_filter_count=1,
            )
        ],
        all_candidates=all_candidates,
    )


def _record(tracker: LeboCSVTracker) -> None:
    tracker.record_iteration(
        iteration_id=1,
        center_sequence="AAA",
        trace=_trace(),
        evaluations=[("ACD", 1.5)],
        candidate_pool_size=1,
        best_sequence="ACD",
        best_objective_value=1.5,
    )


def test_short_stores_only_evaluations(tmp_path: Path) -> None:
    tracker = _tracker(tmp_path, "short")
    _record(tracker)

    assert (tmp_path / "evaluations.csv").exists()
    assert not (tmp_path / "iteration_statistics.csv").exists()
    assert not (tmp_path / "enumeration_steps.csv").exists()
    assert not (tmp_path / "candidates.csv").exists()


def test_normal_stores_step_counts_and_evaluated_provenance(tmp_path: Path) -> None:
    tracker = _tracker(tmp_path, "normal")
    _record(tracker)

    with (tmp_path / "enumeration_steps.csv").open() as input_file:
        steps = list(csv.DictReader(input_file))
    with (tmp_path / "candidates.csv").open() as input_file:
        candidates = list(csv.DictReader(input_file))

    assert steps[0]["proposed_count"] == "20"
    assert [row["sequence"] for row in candidates] == ["ACD"]


def test_all_preserves_every_post_limit_generation_event(tmp_path: Path) -> None:
    tracker = _tracker(tmp_path, "all")
    _record(tracker)

    with (tmp_path / "candidates.csv").open() as input_file:
        candidates = list(csv.DictReader(input_file))

    assert [row["sequence"] for row in candidates] == ["ACD", "ACE"]
    assert candidates[1]["passed_method_filter"] == "False"


def test_legacy_full_level_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="short, normal, or all"):
        _tracker(tmp_path, "full")
