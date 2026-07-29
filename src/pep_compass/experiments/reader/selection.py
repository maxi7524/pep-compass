"""Immutable lazy selections over discovered experiment runs."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from pep_compass.experiments.reader.data_schemas import get_schema
from pep_compass.experiments.reader.entities import ExperimentRun

if TYPE_CHECKING:
    from pep_compass.experiments.reader.reader import ExperimentReader


@dataclass(frozen=True, slots=True)
class ExperimentSelection:
    """Describe selected runs and deferred row filters.

    :param reader: Parent reader owning discovery metadata and metric cache.
    :param runs: Selected run entities.
    :param row_filters: Table-row filters applied only when data is scanned.
    """

    reader: "ExperimentReader"
    runs: tuple[ExperimentRun, ...]
    row_filters: Mapping[str, Any]

    def rows(self, **filters: Any) -> "ExperimentSelection":
        """Return a copy with additional deferred table-row filters."""
        return replace(self, row_filters={**self.row_filters, **filters})

    def select(
        self,
        experiments: list[str] | None = None,
        methods: list[str] | None = None,
        grid_ids: list[str] | None = None,
        peptides: list[str] | None = None,
        seeds: list[int] | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> "ExperimentSelection":
        """Further restrict runs using manifest and resolved-config metadata."""
        selectors = {
            "experiment": experiments,
            "method": methods,
            "grid_id": grid_ids,
            "name": peptides,
            "seed": seeds,
        }
        selected = []
        for run in self.runs:
            metadata = run.metadata
            if any(
                values is not None and metadata.get(key) not in values
                for key, values in selectors.items()
            ):
                continue
            if any(
                metadata.get(key)
                not in (value if isinstance(value, list) else [value])
                for key, value in (parameters or {}).items()
            ):
                continue
            selected.append(run)
        return replace(self, runs=tuple(selected))

    def scan(
        self,
        table: str,
        columns: list[str] | None = None,
        chunk_size: int = 100_000,
    ) -> Iterator[pd.DataFrame]:
        """Yield selected table rows with run metadata attached lazily.

        :param table: Logical table name registered in ``data_schemas``.
        :param columns: Source columns requested by the analysis.
        :param chunk_size: Maximum source rows read at once.
        :return: Iterator of canonical chunks enriched with run metadata.
        """
        schema = get_schema(table)
        for run in self.runs:
            path = run.tracking_path / schema.filename
            if not path.exists():
                continue
            available = schema.inspect(path)
            requested_source = (
                [column for column in columns if column in available]
                if columns is not None
                else None
            )
            requested_metadata = (
                [column for column in columns if column not in available]
                if columns is not None
                else ["experiment", "grid_id", "method", "seed", "name"]
            )
            missing_metadata = set(requested_metadata) - set(run.metadata)
            if missing_metadata:
                raise KeyError(
                    f"Missing run metadata columns: {sorted(missing_metadata)}"
                )
            applicable_filters = {
                key: value
                for key, value in self.row_filters.items()
                if key in available
            }
            for chunk in schema.scan(
                path,
                columns=requested_source,
                filters=applicable_filters,
                chunk_size=chunk_size,
            ):
                for key in requested_metadata:
                    chunk[key] = run.metadata[key]
                if columns is not None:
                    chunk = chunk[columns]
                yield chunk

    def collect(
        self,
        table: str,
        columns: list[str] | None = None,
        chunk_size: int = 100_000,
    ) -> pd.DataFrame:
        """Materialize a selected table when the caller accepts its memory cost."""
        chunks = list(self.scan(table, columns=columns, chunk_size=chunk_size))
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    def paths(self, table: str) -> tuple[Path, ...]:
        """Return existing source paths for one logical table."""
        schema = get_schema(table)
        return tuple(
            path
            for run in self.runs
            if (path := run.tracking_path / schema.filename).exists()
        )

    def specification(self) -> dict[str, Any]:
        """Return stable selection metadata used as an analysis-cache key."""
        return {
            "run_ids": sorted(run.run_id for run in self.runs),
            "tracking_paths": sorted(str(run.tracking_path) for run in self.runs),
            "row_filters": dict(self.row_filters),
        }
