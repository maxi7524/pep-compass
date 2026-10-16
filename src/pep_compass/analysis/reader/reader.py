"""Discovery and lazy access facade for experiment result directories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pep_compass.analysis.reader.dataset import SelectionTableHandle
from pep_compass.analysis.reader.entities import (
    Experiment,
    ExperimentCollection,
    ExperimentRun,
)
from pep_compass.analysis.reader.metrics_store import MetricsStore
from pep_compass.analysis.reader.selection import ExperimentSelection
from pep_compass.data_schemas.dataset import (
    ExperimentDataset,
    RunIdentity,
    RunRecord,
)
from pep_compass.optimization.saving.reader import AnalysisReader


def _flatten(values: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten(value, path))
        elif not isinstance(value, (list, tuple, set)):
            result[path] = value
    return result


class ExperimentReader:
    """Discover canonical run bundles and expose lazy selections over their tables.

    :param path: A run directory or collection of nested run directories.
    :raises FileNotFoundError: If no settings.json/results.sqlite bundle exists.
    """

    def __init__(self, path):
        self.root = Path(path).resolve()
        paths = (
            [self.root]
            if (self.root / "settings.json").is_file()
            else sorted(
                path.parent
                for path in self.root.rglob("settings.json")
                if (path.parent / "results.sqlite").is_file()
            )
        )
        if not paths:
            raise FileNotFoundError(f"No SQLite run bundles below {self.root}.")
        runs, table_names = [], set()
        for path in paths:
            with AnalysisReader(path) as reader:
                config = reader.settings
                run_id = config["run_id"]
                experiment = (
                    path.parent.parent.name
                    if path.parent.name == "runs"
                    else path.parent.name
                )
                metadata = {
                    "run_id": run_id,
                    "experiment": experiment,
                    "grid_id": path.name,
                    "method": "pipeline",
                    "seed": config["runner_settings"]["execution"]["seed"],
                    "name": config["inputs"][0],
                    **_flatten(config),
                }
                runs.append(
                    ExperimentRun(run_id, experiment, path.name, path, config, metadata)
                )
                table_names.update(reader.tables)
        self.collection = ExperimentCollection(
            self.root,
            tuple(
                Experiment(
                    name,
                    self.root,
                    tuple(run for run in runs if run.experiment == name),
                )
                for name in sorted({run.experiment for run in runs})
            ),
        )
        self.metrics = MetricsStore(self.root / ".pep_compass_analysis.sqlite")
        selection = ExperimentSelection(self, self.runs, {})
        self.dataset = ExperimentDataset(
            self.root,
            "sqlite",
            tuple(
                RunRecord(
                    RunIdentity(run.run_id, run.experiment, run.grid_id),
                    run.tracking_path,
                    run.config,
                    run.metadata,
                )
                for run in runs
            ),
            {
                name: SelectionTableHandle(selection, name)
                for name in sorted(table_names)
            },
        )

    @property
    def experiments(self):
        return self.collection.experiments

    @property
    def runs(self):
        return self.collection.runs

    def select(self, **selectors):
        """Create an immutable lazy selection using resolved run metadata."""
        return ExperimentSelection(self, self.runs, {}).select(**selectors)

    def cached_analyses(self):
        """List completed cached analyses without materializing run tables."""
        return self.metrics.list_analyses()

    def replay(self, run_id):
        """Open the canonical reader for one unambiguous run identity."""
        matches = [run for run in self.runs if run.run_id == run_id]
        if len(matches) != 1:
            raise KeyError(run_id)
        return AnalysisReader(matches[0].tracking_path)
