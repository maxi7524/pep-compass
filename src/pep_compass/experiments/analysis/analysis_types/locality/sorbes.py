"""Trajectory-count and step-depth locality analyses for SORBES."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pep_compass.experiments.analysis.analysis_types.locality._streaming import (
    ProgressReporter,
)
from pep_compass.experiments.analysis.result import AnalysisResult
from pep_compass.experiments.reader.selection import ExperimentSelection


def _as_boolean(values: pd.Series) -> pd.Series:
    """Normalize tracker boolean representations."""
    return values.astype("string").str.lower().map(
        {"true": True, "1": True, "false": False, "0": False}
    ).fillna(False)


@dataclass(slots=True)
class CardinalitySketch:
    """Mergeable HyperLogLog sketch for bounded-memory unique counts.

    :param precision: Number of hash bits used as the register index. The
        relative standard error is approximately ``1.04 / sqrt(2**precision)``.
    """

    precision: int = 12
    registers: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 4 <= self.precision <= 18:
            raise ValueError("Sketch precision must be between 4 and 18")
        self.registers = np.zeros(1 << self.precision, dtype=np.uint8)

    def update(self, values: pd.Series) -> None:
        """Add candidate identifiers using a vectorized stable hash."""
        if values.empty:
            return
        hashes = pd.util.hash_array(
            values.astype("string").to_numpy(dtype=object),
            categorize=False,
        ).astype(np.uint64, copy=False)
        mask = np.uint64((1 << self.precision) - 1)
        indices = (hashes & mask).astype(np.intp)
        remaining = hashes >> np.uint64(self.precision)
        exponent = np.frexp(remaining.astype(np.float64))[1]
        width = 64 - self.precision
        ranks = np.where(
            remaining == 0,
            width + 1,
            width - exponent + 1,
        ).astype(np.uint8)
        np.maximum.at(self.registers, indices, ranks)

    def merge(self, other: "CardinalitySketch") -> None:
        """Merge another compatible sketch in place."""
        if other.precision != self.precision:
            raise ValueError("Cannot merge sketches with different precision")
        np.maximum(self.registers, other.registers, out=self.registers)

    def copy(self) -> "CardinalitySketch":
        """Return an independent sketch with the same registers."""
        result = CardinalitySketch(self.precision)
        result.registers[:] = self.registers
        return result

    def estimate(self) -> float:
        """Estimate unique cardinality with the small-range correction."""
        register_count = len(self.registers)
        alpha = 0.7213 / (1.0 + 1.079 / register_count)
        powers = np.exp2(-self.registers.astype(np.float64))
        estimate = alpha * register_count**2 / powers.sum()
        zero_count = int(np.count_nonzero(self.registers == 0))
        if zero_count and estimate <= 2.5 * register_count:
            estimate = register_count * np.log(register_count / zero_count)
        return float(estimate)


def _metadata(run: object) -> dict[str, object]:
    """Return stable run fields needed for cross-peptide comparisons."""
    values = run.metadata
    return {
        "run_id": run.run_id,
        "name": values.get("name"),
        "sequence": values.get("sequence"),
        "experiment": values.get("experiment"),
        "grid_id": values.get("grid_id"),
        "method": values.get("method"),
        "mutation.token_threshold": values.get("mutation.token_threshold"),
    }


def _scan_run_sketches(
    selection: ExperimentSelection,
    radii: tuple[int, ...],
    precision: int,
    chunk_size: int,
    accepted_only: bool,
    progress: bool,
    progress_every_batches: int,
) -> tuple[
    dict[tuple[int, int, int], CardinalitySketch],
    dict[tuple[int, int], set[int]],
]:
    """Scan one run into exact-distance, exact-step cardinality sketches."""
    total = selection.count_rows("candidate_occurrences")
    run_id = selection.runs[0].run_id
    reporter = ProgressReporter(
        f"sorbes_locality:{run_id}",
        total,
        every_batches=progress_every_batches,
        enabled=progress,
    )
    sketches: dict[tuple[int, int, int], CardinalitySketch] = {}
    observed_steps: dict[tuple[int, int], set[int]] = {}
    columns = [
        "iteration_id",
        "trajectory_id",
        "step_id",
        "candidate_id",
        "passed_constraint_filter",
        "levenshtein_to_center",
    ]
    maximum_radius = max(radii)
    for chunk in selection.scan(
        "candidate_occurrences", columns=columns, chunk_size=chunk_size
    ):
        batch_rows = len(chunk)
        chunk["levenshtein_to_center"] = pd.to_numeric(
            chunk["levenshtein_to_center"], errors="coerce"
        )
        chunk["step_id"] = pd.to_numeric(chunk["step_id"], errors="coerce")
        if accepted_only:
            chunk = chunk[_as_boolean(chunk["passed_constraint_filter"])]
        chunk = chunk[
            chunk["levenshtein_to_center"].between(0, maximum_radius)
            & chunk["step_id"].notna()
        ]
        if chunk.empty:
            reporter.update(batch_rows)
            continue
        chunk["levenshtein_to_center"] = chunk[
            "levenshtein_to_center"
        ].astype(int)
        chunk["step_id"] = chunk["step_id"].astype(int)
        group_columns = [
            "iteration_id",
            "trajectory_id",
            "step_id",
            "levenshtein_to_center",
        ]
        for key, group in chunk.groupby(group_columns, sort=False, observed=True):
            iteration, trajectory, step, distance = map(int, key)
            sketch_key = (iteration, trajectory, step, distance)
            sketch = sketches.setdefault(sketch_key, CardinalitySketch(precision))
            sketch.update(group["candidate_id"])
            observed_steps.setdefault((iteration, trajectory), set()).add(step)
        reporter.update(batch_rows)
    return sketches, observed_steps


def _walker_step_index(
    selection: ExperimentSelection,
) -> dict[tuple[int, int], set[int]]:
    """Return complete trajectory-step availability from the compact step table."""
    frame = selection.collect(
        "walker_steps",
        columns=["iteration_id", "trajectory_id", "step_id"],
    )
    result: dict[tuple[int, int], set[int]] = {}
    if frame.empty:
        return result
    for key, group in frame.groupby(
        ["iteration_id", "trajectory_id"], sort=False, observed=True
    ):
        result[tuple(map(int, key))] = set(
            pd.to_numeric(group["step_id"], errors="coerce").dropna().astype(int)
        )
    return result


def _trajectory_radius_sketches(
    sketches: dict[tuple[int, int, int], CardinalitySketch],
    trajectories: list[tuple[int, int]],
    radii: tuple[int, ...],
    precision: int,
) -> dict[int, list[CardinalitySketch]]:
    """Collapse step sketches into one cumulative-radius sketch per trajectory."""
    result = {radius: [] for radius in radii}
    by_trajectory: dict[tuple[int, int], dict[int, CardinalitySketch]] = {
        trajectory: {} for trajectory in trajectories
    }
    for (iteration, trajectory, _step, distance), sketch in sketches.items():
        exact = by_trajectory[(iteration, trajectory)].setdefault(
            distance, CardinalitySketch(precision)
        )
        exact.merge(sketch)
    for trajectory in trajectories:
        cumulative = CardinalitySketch(precision)
        exact = by_trajectory[trajectory]
        previous_radius = 0
        for radius in radii:
            for distance in range(previous_radius + 1, radius + 1):
                if distance in exact:
                    cumulative.merge(exact[distance])
            if 0 in exact and previous_radius == 0:
                cumulative.merge(exact[0])
            result[radius].append(cumulative.copy())
            previous_radius = radius
    return result


def trajectory_saturation(
    selection: ExperimentSelection,
    trajectory_counts: Sequence[int] | None = None,
    levenshtein_radii: Sequence[int] = tuple(range(1, 9)),
    repetitions: int = 300,
    confidence: float = 0.95,
    random_seed: int = 0,
    accepted_only: bool = False,
    sketch_precision: int = 12,
    chunk_size: int = 100_000,
    progress: bool = True,
    progress_every_batches: int = 10,
) -> AnalysisResult:
    """Estimate local candidate yield under random trajectory subsets.

    :param selection: Runs and deferred row filters to analyze.
    :param trajectory_counts: Numbers of complete trajectories sampled per run.
    :param levenshtein_radii: Cumulative sequence-locality radii.
    :param repetitions: Random subsets evaluated for each non-complete count.
    :param confidence: Percentile interval coverage within each run.
    :param random_seed: Deterministic subset seed.
    :param accepted_only: Restrict input to the configured constraint filter.
        ``False`` analyzes the complete post-cap pool tracked in long mode.
    :param sketch_precision: HyperLogLog precision controlling count error/RAM.
    :param chunk_size: Maximum occurrence rows read at once.
    :param progress: Print scan progress and ETA.
    :param progress_every_batches: Batches between progress messages.
    :return: Per-peptide rarefaction curves for cumulative edit-distance radii.
    """
    if repetitions < 1 or chunk_size < 1 or progress_every_batches < 1:
        raise ValueError("Repetitions and processing bounds must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    radii = tuple(sorted({int(radius) for radius in levenshtein_radii if radius > 0}))
    if not radii:
        raise ValueError("At least one positive Levenshtein radius is required")
    defaults = (1, 2, 5, 10, 20, 30, 50, 75, 100)
    rows: list[dict[str, object]] = []
    tail = (1.0 - confidence) / 2.0
    for run_index, run in enumerate(selection.runs):
        run_selection = ExperimentSelection(
            selection.reader, (run,), selection.row_filters
        )
        sketches, _ = _scan_run_sketches(
            run_selection,
            radii,
            sketch_precision,
            chunk_size,
            accepted_only,
            progress,
            progress_every_batches,
        )
        observed_steps = _walker_step_index(run_selection)
        trajectories = sorted(observed_steps)
        if not trajectories:
            continue
        radius_sketches = _trajectory_radius_sketches(
            sketches, trajectories, radii, sketch_precision
        )
        maximum = len(trajectories)
        requested = trajectory_counts or defaults
        counts = sorted({int(value) for value in requested if 0 < value <= maximum})
        metadata = _metadata(run)
        rng = np.random.default_rng(random_seed + run_index)
        subsets = {
            count: [
                (
                    np.arange(maximum)
                    if count == maximum
                    else rng.choice(maximum, size=count, replace=False)
                )
                for _ in range(1 if count == maximum else repetitions)
            ]
            for count in counts
        }
        for radius in radii:
            registers = np.stack(
                [sketch.registers for sketch in radius_sketches[radius]]
            )
            for count in counts:
                estimates = []
                for selected in subsets[count]:
                    merged = CardinalitySketch(sketch_precision)
                    merged.registers[:] = registers[selected].max(axis=0)
                    estimates.append(merged.estimate())
                values = np.asarray(estimates)
                rows.append(
                    {
                        **metadata,
                        "levenshtein_radius": radius,
                        "trajectory_count": count,
                        "available_trajectories": maximum,
                        "unique_candidates": float(values.mean()),
                        "subset_confidence_low": float(np.quantile(values, tail)),
                        "subset_confidence_high": float(
                            np.quantile(values, 1.0 - tail)
                        ),
                    }
                )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            ["run_id", "levenshtein_radius", "trajectory_count"]
        )
        grouped = result.groupby(["run_id", "levenshtein_radius"])
        result["marginal_unique_candidates"] = grouped[
            "unique_candidates"
        ].diff()
        result["marginal_per_trajectory"] = result[
            "marginal_unique_candidates"
        ].div(grouped["trajectory_count"].diff())
    return AnalysisResult(
        result.reset_index(drop=True),
        {
            "analysis": "trajectory_saturation",
            "unit": "peptide run",
            "radii_are_cumulative": True,
            "accepted_only": accepted_only,
            "cardinality_method": "HyperLogLog",
            "cardinality_relative_standard_error": 1.04
            / np.sqrt(2**sketch_precision),
            "resampling": "trajectory subsets without replacement",
        },
    )


def step_saturation(
    selection: ExperimentSelection,
    max_steps: Sequence[int] | None = None,
    levenshtein_radii: Sequence[int] = tuple(range(1, 9)),
    accepted_only: bool = False,
    sketch_precision: int = 12,
    chunk_size: int = 100_000,
    progress: bool = True,
    progress_every_batches: int = 10,
) -> AnalysisResult:
    """Calculate exact central depth curves for every peptide and radius.

    :param selection: Runs and deferred row filters to analyze.
    :param max_steps: Walker-depth cutoffs; all observed steps are used by default.
    :param levenshtein_radii: Cumulative sequence-locality radii.
    :param accepted_only: Restrict input to the configured constraint filter.
    :param sketch_precision: HyperLogLog precision controlling count error/RAM.
    :param chunk_size: Maximum occurrence rows read at once.
    :param progress: Print scan progress and ETA.
    :param progress_every_batches: Batches between progress messages.
    :return: Per-peptide cumulative and marginal candidate depth curves.
    """
    if chunk_size < 1 or progress_every_batches < 1:
        raise ValueError("Processing bounds must be positive")
    radii = tuple(sorted({int(radius) for radius in levenshtein_radii if radius > 0}))
    if not radii:
        raise ValueError("At least one positive Levenshtein radius is required")
    rows: list[dict[str, object]] = []
    for run in selection.runs:
        run_selection = ExperimentSelection(
            selection.reader, (run,), selection.row_filters
        )
        sketches, _ = _scan_run_sketches(
            run_selection,
            radii,
            sketch_precision,
            chunk_size,
            accepted_only,
            progress,
            progress_every_batches,
        )
        observed_steps = _walker_step_index(run_selection)
        trajectories = sorted(observed_steps)
        if not trajectories:
            continue
        available = sorted(
            {step for steps in observed_steps.values() for step in steps}
        )
        selected_steps = sorted(
            set(available if max_steps is None else map(int, max_steps))
            & set(available)
        )
        metadata = _metadata(run)
        cumulative = {
            trajectory: {
                radius: CardinalitySketch(sketch_precision) for radius in radii
            }
            for trajectory in trajectories
        }
        previous_counts = {radius: 0.0 for radius in radii}
        for step in available:
            for trajectory in trajectories:
                iteration, trajectory_id = trajectory
                for radius in radii:
                    for distance in range(0, radius + 1):
                        sketch = sketches.get(
                            (iteration, trajectory_id, step, distance)
                        )
                        if sketch is not None:
                            cumulative[trajectory][radius].merge(sketch)
            if step not in selected_steps:
                continue
            active = sum(
                step in observed_steps[trajectory] for trajectory in trajectories
            )
            for radius in radii:
                union = CardinalitySketch(sketch_precision)
                per_trajectory = []
                for trajectory in trajectories:
                    sketch = cumulative[trajectory][radius]
                    union.merge(sketch)
                    per_trajectory.append(sketch.estimate())
                count = union.estimate()
                new_count = max(0.0, count - previous_counts[radius])
                previous_counts[radius] = count
                rows.append(
                    {
                        **metadata,
                        "levenshtein_radius": radius,
                        "max_step": step,
                        "trajectory_count": len(trajectories),
                        "active_trajectories": active,
                        "active_trajectory_fraction": active / len(trajectories),
                        "unique_candidates": count,
                        "new_unique_candidates": new_count,
                        "new_candidates_per_active_trajectory": (
                            new_count / active if active else np.nan
                        ),
                        "per_trajectory_candidates_median": float(
                            np.median(per_trajectory)
                        ),
                    }
                )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            ["run_id", "levenshtein_radius", "max_step"]
        ).reset_index(drop=True)
    return AnalysisResult(
        result,
        {
            "analysis": "step_saturation",
            "unit": "peptide run",
            "radii_are_cumulative": True,
            "accepted_only": accepted_only,
            "central_curve": "complete observed trajectory union",
            "cardinality_method": "HyperLogLog",
            "cardinality_relative_standard_error": 1.04
            / np.sqrt(2**sketch_precision),
        },
    )
