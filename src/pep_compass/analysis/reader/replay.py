"""Typed access to replay checkpoints written by the optimization runtime."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch


@dataclass(frozen=True, slots=True)
class ReplayVerification:
    """Report whether reconstructed local-enumeration output matches a checkpoint."""

    execution_id: int
    expected_count: int
    actual_count: int
    expected_sha256: str
    actual_sha256: str

    @property
    def matches(self) -> bool:
        """Return whether both ordered sequence count and digest match."""
        return (
            self.expected_count == self.actual_count
            and self.expected_sha256 == self.actual_sha256
        )


class RunReplay:
    """Read final results and minimal checkpoints for one materialized run."""

    def __init__(self, run_directory: str | Path) -> None:
        self.root = Path(run_directory)
        self.tracking = self.root / "tracking"
        if not (self.root / "result.json").is_file():
            raise FileNotFoundError(self.root / "result.json")

    @property
    def result(self) -> dict[str, Any]:
        """Return terminal run metadata."""
        return json.loads((self.root / "result.json").read_text(encoding="utf-8"))

    @property
    def manifest(self) -> dict[str, Any]:
        """Return runtime and replay-contract metadata."""
        path = self.tracking / "replay_manifest.json"
        return json.loads(path.read_text(encoding="utf-8"))

    @property
    def resolved_configuration(self) -> dict[str, Any]:
        """Return the exact resolved configuration persisted for this run."""
        return json.loads(
            (self.root / "resolved_config.json").read_text(encoding="utf-8")
        )

    @property
    def final_candidates(self) -> pd.DataFrame:
        """Return final sequences and all scalar oracle scores."""
        return pd.read_csv(self.root / "candidates.csv")

    @property
    def final_latents(self) -> torch.Tensor:
        """Return final candidate latent origins shaped ``(B, D)``."""
        return torch.load(
            self.root / "latent_origins.pt",
            map_location="cpu",
            weights_only=True,
        )

    @property
    def local_enumerations(self) -> pd.DataFrame:
        """Return local-enumeration replay boundaries and output digests."""
        return pd.read_csv(self.tracking / "local_enumerations.csv")

    @property
    def trajectory_points(self) -> pd.DataFrame:
        """Return SORBES point metadata indexed into :attr:`trajectory_latents`."""
        return pd.read_csv(self.tracking / "trajectory_points.csv")

    @property
    def trajectory_latents(self) -> torch.Tensor:
        """Return SORBES point latents shaped ``(N, D)``."""
        return torch.load(
            self.tracking / "trajectory_latents.pt",
            map_location="cpu",
            weights_only=True,
        )

    def trajectory(self, trajectory_id: str) -> tuple[pd.DataFrame, torch.Tensor]:
        """Return ordered metadata and latent points for one trajectory."""
        points = self.trajectory_points
        points = points[points["trajectory_id"] == trajectory_id].sort_values(
            ["trajectory_step", "execution_id"]
        )
        indices = torch.as_tensor(points["latent_index"].to_numpy(), dtype=torch.long)
        return points.reset_index(drop=True), self.trajectory_latents.index_select(
            0, indices
        )

    def local_enumeration_input(
        self,
        execution_id: int,
    ) -> tuple[tuple[str, ...], torch.Tensor]:
        """Return sequences and latent checkpoint used by one local enumeration."""
        rows = self.local_enumerations
        matched = rows[rows["execution_id"] == execution_id]
        if len(matched) != 1:
            raise KeyError(f"Unknown local-enumeration execution: {execution_id}")
        row = matched.iloc[0]
        start = int(row["input_latent_start"])
        count = int(row["input_latent_count"])
        latents = torch.load(
            self.tracking / "local_enumeration_inputs.pt",
            map_location="cpu",
            weights_only=True,
        )
        sequences = tuple(json.loads(row["input_sequences"]))
        return sequences, latents[start : start + count]

    def verify_local_enumeration(
        self,
        execution_id: int,
        reconstructed_sequences: Sequence[str],
    ) -> ReplayVerification:
        """Compare replayed ordered sequences with the stored result checkpoint."""
        rows = self.local_enumerations
        matched = rows[rows["execution_id"] == execution_id]
        if len(matched) != 1:
            raise KeyError(f"Unknown local-enumeration execution: {execution_id}")
        row = matched.iloc[0]
        actual = list(reconstructed_sequences)
        payload = json.dumps(
            actual,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return ReplayVerification(
            execution_id=execution_id,
            expected_count=int(row["output_count"]),
            actual_count=len(actual),
            expected_sha256=str(row["output_sequences_sha256"]),
            actual_sha256=hashlib.sha256(payload).hexdigest(),
        )
