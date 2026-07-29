"""Discovery and lazy access facade for experiment result directories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from pep_compass.experiments.reader.entities import (
    Experiment,
    ExperimentCollection,
    ExperimentRun,
)
from pep_compass.experiments.reader.metrics_store import MetricsStore
from pep_compass.experiments.reader.selection import ExperimentSelection


def _flatten(values: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten(value, path))
        elif not isinstance(value, (list, tuple, set)):
            result[path] = value
    return result


def _repository_root(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd()


def _resolve_path(value: str | Path, *roots: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    for root in roots:
        candidate = (root / path).resolve()
        if candidate.exists():
            return candidate
    return (roots[0] / path).resolve()


class ExperimentReader:
    """Discover experiment outputs and expose lazy selections over their tables."""

    def __init__(self, path: str | Path) -> None:
        """Open a collection, experiment directory, or individual tracking run.

        :param path: Result path at any supported hierarchy level.
        :raises FileNotFoundError: If no experiment outputs are discovered.
        """
        self.root = Path(path).resolve()
        if not self.root.exists():
            raise FileNotFoundError(self.root)
        self.repository_root = _repository_root(self.root)
        self.collection = self._discover()
        self.metrics = MetricsStore(self.root / ".pep_compass_analysis.sqlite")

    @property
    def experiments(self) -> tuple[Experiment, ...]:
        """Return discovered experiments without loading tracking tables."""
        return self.collection.experiments

    @property
    def runs(self) -> tuple[ExperimentRun, ...]:
        """Return all discovered run entities."""
        return self.collection.runs

    def select(self, **selectors: Any) -> ExperimentSelection:
        """Create an immutable lazy selection over discovered runs."""
        return ExperimentSelection(self, self.runs, {}).select(**selectors)

    def cached_analyses(self) -> pd.DataFrame:
        """List completed cached results, including cache-only collections."""
        return self.metrics.list_analyses()

    def _discover(self) -> ExperimentCollection:
        if (self.root / "tracking_metadata.json").exists():
            run = self._tracking_run(self.root)
            experiment = Experiment(self.root.parent.name, self.root, (run,))
            return ExperimentCollection(self.root, (experiment,))
        experiment_roots = []
        if (self.root / "run_manifest.csv").exists():
            experiment_roots = [self.root]
        else:
            experiment_roots = sorted(
                path.parent for path in self.root.glob("*/run_manifest.csv")
            )
        if not experiment_roots:
            if (self.root / ".pep_compass_analysis.sqlite").exists():
                return ExperimentCollection(self.root, ())
            raise FileNotFoundError(
                f"No tracking_metadata.json or run_manifest.csv below {self.root}"
            )
        experiments = tuple(self._manifest_experiment(root) for root in experiment_roots)
        return ExperimentCollection(self.root, experiments)

    def _tracking_run(self, tracking_path: Path) -> ExperimentRun:
        with (tracking_path / "tracking_metadata.json").open() as handle:
            metadata = json.load(handle)
        resolved_config_path = tracking_path.parent.parent / "resolved_config.json"
        config = {}
        if resolved_config_path.exists():
            with resolved_config_path.open(encoding="utf-8") as handle:
                config = json.load(handle)
        run_id = str(metadata["run_id"])
        values = {
            "run_id": run_id,
            "experiment": tracking_path.parent.name,
            "grid_id": tracking_path.parent.parent.name,
            "method": metadata.get("candidate_strategy"),
            "tracking_level": metadata.get("tracking_level"),
            "seed": config.get("seed"),
            "name": None,
            "mutation.token_threshold": config.get("mutation", {}).get(
                "token_threshold"
            ),
            **_flatten(config),
        }
        return ExperimentRun(
            run_id=run_id,
            experiment=values["experiment"],
            grid_id=values["grid_id"],
            tracking_path=tracking_path,
            config=config,
            metadata=values,
        )

    def _manifest_experiment(self, root: Path) -> Experiment:
        runs_frame = pd.read_csv(root / "run_manifest.csv")
        grid_parameters: dict[str, dict[str, Any]] = {}
        grid_manifest = root / "grid_manifest.csv"
        if grid_manifest.exists():
            for row in pd.read_csv(grid_manifest).to_dict(orient="records"):
                grid_parameters[str(row["grid_id"])] = json.loads(row["parameters"])
        runs = []
        for row in runs_frame.to_dict(orient="records"):
            task_path = _resolve_path(
                row["task_file"], self.repository_root, root, self.root
            )
            with task_path.open(encoding="utf-8") as handle:
                task = json.load(handle)
            config = task["config"]
            output_path = _resolve_path(
                task["output_path"], self.repository_root, root, self.root
            )
            tracking_path = output_path / "tracking" / task["experiment_id"]
            metadata = {
                **row,
                "run_id": task["task_id"],
                "experiment": root.name,
                "experiment_id": task["experiment_id"],
                "method": config["optimizer"].get("lebo", {}).get(
                    "candidate_strategy", config["optimizer"]["name"]
                ),
                "tracking_level": config.get("tracking", {}).get("level"),
                **_flatten(config),
                **grid_parameters.get(str(row["grid_id"]), {}),
            }
            runs.append(
                ExperimentRun(
                    run_id=task["task_id"],
                    experiment=root.name,
                    grid_id=str(row["grid_id"]),
                    tracking_path=tracking_path,
                    config=config,
                    metadata=metadata,
                )
            )
        return Experiment(root.name, root, tuple(runs))
