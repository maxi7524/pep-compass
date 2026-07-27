"""Manifest-backed catalog of locality experiment runs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _resolve_manifest_path(path: str, repository_root: Path) -> Path:
    """Resolve an absolute or repository-relative manifest path."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repository_root / candidate


class LocalityExperiment:
    """Index experiment grids and runs without loading tracking tables.

    :param results_root: Directory containing experiment result directories.
    :param catalog: One row per materialized optimization task.
    :param repository_root: Repository root used to resolve relative paths.
    """

    def __init__(
        self,
        results_root: Path,
        catalog: pd.DataFrame,
        repository_root: Path,
    ) -> None:
        self.results_root = results_root
        self.catalog = catalog
        self.repository_root = repository_root

    @classmethod
    def open(
        cls,
        results_root: str | Path,
        repository_root: str | Path | None = None,
    ) -> "LocalityExperiment":
        """Build a lightweight run catalog from grid and run manifests.

        :param results_root: Directory containing one or more experiment roots.
        :param repository_root: Root for repository-relative manifest paths.
            When omitted, it is inferred from ``results/locality`` layout.
        :return: Manifest-backed experiment catalog.
        :raises FileNotFoundError: If no run manifests are found.
        """
        root = Path(results_root).resolve()
        repo = (
            Path(repository_root).resolve()
            if repository_root is not None
            else root.parent.parent
        )
        run_manifests = sorted(root.glob("*/run_manifest.csv"))
        if not run_manifests and (root / "run_manifest.csv").exists():
            run_manifests = [root / "run_manifest.csv"]
        if not run_manifests:
            raise FileNotFoundError(f"No run_manifest.csv files found below {root}")

        rows: list[dict[str, Any]] = []
        for run_manifest in run_manifests:
            experiment_root = run_manifest.parent
            grid_manifest = pd.read_csv(experiment_root / "grid_manifest.csv")
            grid_parameters = {
                row.grid_id: json.loads(row.parameters)
                for row in grid_manifest.itertuples(index=False)
            }
            runs = pd.read_csv(run_manifest)
            logger.info("Indexing %s runs from %s", len(runs), experiment_root)
            for run in runs.to_dict(orient="records"):
                task_path = _resolve_manifest_path(run["task_file"], repo)
                with task_path.open(encoding="utf-8") as input_file:
                    task = json.load(input_file)
                config = task["config"]
                output_path = _resolve_manifest_path(task["output_path"], repo)
                tracking_path = output_path / "tracking" / task["experiment_id"]
                row = {
                    **run,
                    "experiment": experiment_root.name,
                    "experiment_root": str(experiment_root),
                    "task_path": str(task_path),
                    "tracking_path": str(tracking_path),
                    "experiment_id": task["experiment_id"],
                    "method": config["optimizer"].get("lebo", {}).get(
                        "candidate_strategy", config["optimizer"]["name"]
                    ),
                    "tracking_level": config.get("tracking", {}).get("level"),
                    "config": config,
                }
                for parameter, value in grid_parameters[run["grid_id"]].items():
                    row[parameter] = value
                rows.append(row)
        return cls(root, pd.DataFrame(rows), repo)

    def select(
        self,
        experiments: list[str] | None = None,
        methods: list[str] | None = None,
        grid_ids: list[str] | None = None,
        peptides: list[str] | None = None,
        repetitions: list[int] | None = None,
        parameters: dict[str, list[Any] | Any] | None = None,
        iteration_min: int | None = None,
        iteration_max: int | None = None,
    ):
        """Select runs and deferred row filters for subsequent analysis."""
        from pep_compass.experiments.analysis.selection import ExperimentSelection

        frame = self.catalog
        selectors = {
            "experiment": experiments,
            "method": methods,
            "grid_id": grid_ids,
            "name": peptides,
            "repetition": repetitions,
        }
        for column, values in selectors.items():
            if values is not None:
                frame = frame[frame[column].isin(values)]
        for column, values in (parameters or {}).items():
            accepted = values if isinstance(values, list) else [values]
            if column not in frame.columns:
                raise KeyError(f"Grid parameter is not present in catalog: {column}")
            frame = frame[frame[column].isin(accepted)]
        return ExperimentSelection(
            experiment=self,
            runs=frame.reset_index(drop=True),
            iteration_min=iteration_min,
            iteration_max=iteration_max,
        )
