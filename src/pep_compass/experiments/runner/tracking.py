"""Helpers controlling when detailed experiment tracking becomes active."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def should_start_tracking(
    tracking: Mapping[str, Any], iteration_id: int, center_sequence: str
) -> bool:
    """Return whether either configured tracking threshold has been reached.

    :param tracking: Tracking configuration containing optional thresholds.
    :param iteration_id: Current zero-based optimizer iteration.
    :param center_sequence: Current enumeration centre sequence.
    :return: Whether tracking should be active. With no thresholds configured,
        tracking starts immediately.
    """
    start_iteration = tracking.get("start_iteration")
    start_length = tracking.get("start_sequence_length")
    if start_iteration is None and start_length is None:
        return True
    return bool(
        (start_iteration is not None and iteration_id >= start_iteration)
        or (start_length is not None and len(center_sequence) >= start_length)
    )
