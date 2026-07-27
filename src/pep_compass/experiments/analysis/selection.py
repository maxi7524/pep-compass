"""Deferred, chunked access to selected experiment tracking tables."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

if TYPE_CHECKING:
    from pep_compass.experiments.analysis.catalog import LocalityExperiment

logger = logging.getLogger(__name__)

TrackingSource = Literal["steps", "candidates", "evaluations", "iterations"]
SOURCE_FILES: dict[TrackingSource, str] = {
    "steps": "enumeration_steps.csv",
    "candidates": "candidates.csv",
    "evaluations": "evaluations.csv",
    "iterations": "iteration_statistics.csv",
}


@dataclass(frozen=True)
class ExperimentSelection:
    """A run selection whose large CSV files remain unloaded.

    :param experiment: Parent manifest catalog.
    :param runs: Selected catalog rows.
    :param iteration_min: Inclusive minimum optimizer iteration.
    :param iteration_max: Inclusive maximum optimizer iteration.
    :param trajectory_ids: Optional trajectory identifiers retained in scans.
    :param step_min: Inclusive minimum walker step.
    :param step_max: Inclusive maximum walker step.
    """

    experiment: "LocalityExperiment"
    runs: pd.DataFrame
    iteration_min: int | None = None
    iteration_max: int | None = None
    trajectory_ids: tuple[int, ...] | None = None
    step_min: int | None = None
    step_max: int | None = None

    def rows(
        self,
        trajectory_ids: list[int] | None = None,
        step_min: int | None = None,
        step_max: int | None = None,
    ) -> "ExperimentSelection":
        """Return a copy with deferred trajectory and step row filters."""
        return replace(
            self,
            trajectory_ids=(
                tuple(trajectory_ids) if trajectory_ids is not None else None
            ),
            step_min=step_min,
            step_max=step_max,
        )

    def _filter_rows(self, frame: pd.DataFrame) -> pd.DataFrame:
        if "iteration_id" in frame:
            if self.iteration_min is not None:
                frame = frame[frame["iteration_id"] >= self.iteration_min]
            if self.iteration_max is not None:
                frame = frame[frame["iteration_id"] <= self.iteration_max]
        if self.trajectory_ids is not None and "trajectory_id" in frame:
            frame = frame[frame["trajectory_id"].isin(self.trajectory_ids)]
        if "step_id" in frame:
            if self.step_min is not None:
                frame = frame[frame["step_id"] >= self.step_min]
            if self.step_max is not None:
                frame = frame[frame["step_id"] <= self.step_max]
        return frame

    def scan(
        self,
        source: TrackingSource,
        columns: list[str] | None = None,
        chunk_size: int = 100_000,
    ) -> Iterator[pd.DataFrame]:
        """Yield filtered tracking rows in bounded-memory chunks.

        :param source: Logical tracking table to scan.
        :param columns: Optional source columns to read.
        :param chunk_size: Maximum CSV rows loaded per chunk.
        :return: Iterator of selected chunks enriched with run/grid metadata.
        """
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        filename = SOURCE_FILES[source]
        filter_columns = {"iteration_id", "trajectory_id", "step_id"}
        requested_columns = list(columns) if columns is not None else None
        metadata_columns = [
            column
            for column in self.runs.columns
            if column not in {"config"}
        ]
        for run in self.runs.to_dict(orient="records"):
            path = Path(run["tracking_path"]) / filename
            if not path.exists():
                logger.debug("Skipping missing tracking table: %s", path)
                continue
            available = pd.read_csv(path, nrows=0).columns.tolist()
            read_columns = None
            if requested_columns is not None:
                missing = set(requested_columns) - set(available)
                if missing:
                    raise KeyError(f"Missing columns in {path}: {sorted(missing)}")
                read_columns = list(
                    dict.fromkeys(
                        requested_columns
                        + [name for name in filter_columns if name in available]
                    )
                )
            for chunk in pd.read_csv(
                path, usecols=read_columns, chunksize=chunk_size
            ):
                chunk = self._filter_rows(chunk)
                if chunk.empty:
                    continue
                if requested_columns is not None:
                    chunk = chunk[requested_columns]
                for column in metadata_columns:
                    if column not in chunk:
                        chunk[column] = run[column]
                yield chunk

    def collect(
        self,
        source: TrackingSource,
        columns: list[str] | None = None,
        chunk_size: int = 100_000,
    ) -> pd.DataFrame:
        """Materialize a selected table when the caller accepts its memory cost."""
        chunks = list(self.scan(source, columns=columns, chunk_size=chunk_size))
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    def apply_batches(
        self,
        source: TrackingSource,
        operation: Callable[[pd.DataFrame], Any],
        columns: list[str] | None = None,
        chunk_size: int = 100_000,
    ) -> Iterator[Any]:
        """Apply an arbitrary operation independently to every selected chunk."""
        for chunk in self.scan(source, columns=columns, chunk_size=chunk_size):
            yield operation(chunk)
