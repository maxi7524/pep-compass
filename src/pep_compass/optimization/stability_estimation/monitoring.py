"""Efficient process and CUDA memory sampling at execution boundaries."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import torch

from pep_compass.data.optimization import CandidateBatch
from pep_compass.optimization.stability_estimation.estimation import (
    estimate_batch_memory,
)
from pep_compass.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    """Store one low-overhead memory observation at a step boundary."""

    label: str
    candidates: int
    batch_bytes: int
    rss_bytes: int | None
    cuda_allocated_bytes: int | None
    cuda_reserved_bytes: int | None
    cuda_peak_bytes: int | None


class NullStabilityMonitor:
    """Disable resource sampling without branching in engine callers."""

    def sample(self, label: str, batch: CandidateBatch) -> MemorySnapshot | None:
        """Return no snapshot for a disabled monitor."""
        return None


class StabilityMonitor:
    """Sample and log batch, process and CUDA memory at step boundaries.

    :param enabled: Disable all sampling while retaining the same interface.
    :param log_level: Logging level used for emitted snapshots.
    """

    def __init__(self, *, enabled: bool = True, log_level: int = 10) -> None:
        self.enabled = enabled
        self.log_level = log_level
        self.snapshots: list[MemorySnapshot] = []

    def sample(self, label: str, batch: CandidateBatch) -> MemorySnapshot | None:
        """Collect and log one memory snapshot.

        :param label: Stable execution-boundary label.
        :param batch: Candidate batch live at the boundary.
        :return: Collected snapshot, or ``None`` when monitoring is disabled.
        """
        if not self.enabled:
            return None
        estimate = estimate_batch_memory(batch)
        snapshot = MemorySnapshot(
            label=label,
            candidates=len(batch),
            batch_bytes=estimate.total_bytes,
            rss_bytes=_current_rss_bytes(),
            cuda_allocated_bytes=_cuda_value(torch.cuda.memory_allocated),
            cuda_reserved_bytes=_cuda_value(torch.cuda.memory_reserved),
            cuda_peak_bytes=_cuda_value(torch.cuda.max_memory_allocated),
        )
        self.snapshots.append(snapshot)
        logger.log(
            self.log_level,
            "Stability label=%s candidates=%s batch_bytes=%s rss_bytes=%s "
            "cuda_allocated_bytes=%s cuda_reserved_bytes=%s cuda_peak_bytes=%s.",
            snapshot.label,
            snapshot.candidates,
            snapshot.batch_bytes,
            snapshot.rss_bytes,
            snapshot.cuda_allocated_bytes,
            snapshot.cuda_reserved_bytes,
            snapshot.cuda_peak_bytes,
        )
        return snapshot


def _current_rss_bytes() -> int | None:
    """Read current resident memory on Linux without an external dependency."""
    statm = Path("/proc/self/statm")
    if not statm.exists():
        return None
    try:
        resident_pages = int(statm.read_text(encoding="ascii").split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (IndexError, OSError, ValueError):
        return None


def _cuda_value(function) -> int | None:
    """Return one CUDA allocator counter without forcing device creation."""
    if not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return None
    return int(function())
