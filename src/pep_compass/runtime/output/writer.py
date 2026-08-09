"""Atomic output of run results and stability measurements."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from threading import Lock
from typing import Any

import torch

from pep_compass.data.result_schema import CURRENT_RESULT_SCHEMA_VERSION
from pep_compass.optimization.engine.execution.result import OptimizationResult
from pep_compass.optimization.stability_estimation.monitoring import MemorySnapshot
from pep_compass.runtime.planning.plan import PlannedRun


class ResultWriter:
    """Write independently recoverable run results below one output root."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def run_directory(self, entry: PlannedRun) -> Path:
        """Return the stable output directory for a plan entry."""
        return self.root / "variants" / entry.variant.variant_id / "runs" / entry.run_id

    def is_completed(self, entry: PlannedRun) -> bool:
        """Return whether a plan entry has a durable completed status."""
        status_path = self.run_directory(entry) / "result.json"
        if not status_path.exists():
            return False
        try:
            return json.loads(status_path.read_text(encoding="utf-8"))["status"] == "completed"
        except (KeyError, json.JSONDecodeError):
            return False

    def write_running(self, entry: PlannedRun) -> None:
        """Persist a running status before computation starts."""
        self._write_status(entry, "running")

    def write_completed(
        self,
        entry: PlannedRun,
        result: OptimizationResult,
        snapshots: list[MemorySnapshot],
    ) -> None:
        """Persist candidates, stability measurements and completed status."""
        directory = self.run_directory(entry)
        directory.mkdir(parents=True, exist_ok=True)
        self._write_candidates(directory / "candidates.csv", result)
        torch.save(result.candidates.latent_origins.detach().cpu(), directory / "latent_origins.pt")
        self._write_stability(directory / "stability.csv", snapshots)
        self._write_status(
            entry,
            "completed",
            candidate_count=len(result.candidates),
            best_score=result.best_score,
            objective_name=result.objective_name,
            objective_direction=result.objective_direction,
        )

    def write_failed(self, entry: PlannedRun, error: BaseException) -> None:
        """Persist a failed status and concise exception information."""
        self._write_status(
            entry,
            "failed",
            error=f"{type(error).__name__}: {error}",
        )

    def _write_status(self, entry: PlannedRun, status: str, **values: Any) -> None:
        """Atomically replace one run status document."""
        directory = self.run_directory(entry)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": CURRENT_RESULT_SCHEMA_VERSION,
            "status": status,
            "run_id": entry.run_id,
            "run_index": entry.index,
            "task_id": entry.task.task_id,
            "variant_id": entry.variant.variant_id,
            "seed": entry.seed,
            "sequence": entry.task.sequence,
            **values,
        }
        target = directory / "result.json"
        temporary = directory / "result.json.tmp"
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(target)

    @staticmethod
    def _write_candidates(path: Path, result: OptimizationResult) -> None:
        """Write final candidate sequences and optional objective scores."""
        score_field = None
        for name, field in result.candidates.fields.items():
            if name.startswith("oracle.") and name.endswith(".score"):
                score_field = getattr(field, "values", None)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("candidate_index", "sequence", "score"))
            writer.writeheader()
            for index, sequence in enumerate(result.candidates.sequences):
                score = "" if score_field is None else float(score_field[index].item())
                writer.writerow({"candidate_index": index, "sequence": sequence, "score": score})

    @staticmethod
    def _write_stability(path: Path, snapshots: list[MemorySnapshot]) -> None:
        """Write low-overhead runtime memory measurements."""
        fields = tuple(MemorySnapshot.__dataclass_fields__)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for snapshot in snapshots:
                writer.writerow({field: getattr(snapshot, field) for field in fields})
