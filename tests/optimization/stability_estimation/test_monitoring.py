"""Tests for low-overhead runtime stability snapshots."""

from pep_compass.optimization.stability_estimation import StabilityMonitor
from fixtures.candidates import candidate_batch


def test_monitor_records_batch_memory_without_requiring_cuda() -> None:
    """CPU monitoring must return a useful snapshot on systems without CUDA."""
    monitor = StabilityMonitor()

    snapshot = monitor.sample("test", candidate_batch())

    assert snapshot is not None
    assert snapshot.label == "test"
    assert snapshot.candidates == 2
    assert snapshot.batch_bytes > 0
