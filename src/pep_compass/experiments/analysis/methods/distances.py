"""Sequence and encoder-derived locality distances."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import Levenshtein
import numpy as np
import pandas as pd

from pep_compass.experiments.analysis.selection import ExperimentSelection


def _hamming(left: str, right: str) -> int | None:
    if len(left) != len(right):
        return None
    return sum(a != b for a, b in zip(left, right))


def add_sequence_distances(
    frame: pd.DataFrame,
    reference_column: str = "parent_sequence",
) -> pd.DataFrame:
    """Add Hamming and Levenshtein distances to a candidate frame."""
    result = frame.copy()
    pairs = zip(result["sequence"], result[reference_column])
    pairs = list(pairs)
    result["hamming_distance"] = [_hamming(left, right) for left, right in pairs]
    result["levenshtein_distance"] = [
        Levenshtein.distance(left, right) for left, right in pairs
    ]
    return result


def latent_distance_batches(
    selection: ExperimentSelection,
    encoder_decoder: Any,
    chunk_size: int = 4096,
) -> Iterator[pd.DataFrame]:
    """Encode candidate-parent pairs and yield Euclidean latent distances.

    :param selection: Candidate selection scanned lazily.
    :param encoder_decoder: Object exposing ``encode_peptides``.
    :param chunk_size: Maximum candidate rows encoded in one batch.
    :return: Candidate chunks with sequence and latent-space distances.
    """
    columns = ["sequence", "parent_sequence", "iteration_id", "trajectory_id", "step_id"]
    for chunk in selection.scan("candidates", columns=columns, chunk_size=chunk_size):
        candidate_latents = encoder_decoder.encode_peptides(chunk["sequence"].tolist())
        parent_latents = encoder_decoder.encode_peptides(
            chunk["parent_sequence"].tolist()
        )
        distances = np.linalg.norm(
            candidate_latents.detach().cpu().numpy()
            - parent_latents.detach().cpu().numpy(),
            axis=1,
        )
        result = add_sequence_distances(chunk)
        result["latent_euclidean_distance"] = distances
        yield result
