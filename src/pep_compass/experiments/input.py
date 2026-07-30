"""Input sequence loading for composable experiments."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def load_input_sequences(
    configuration: Mapping[str, Any],
    *,
    base_directory: Path | None = None,
) -> list[str]:
    """Load starting sequences from an inline list or CSV file.

    Exactly one of ``sequences`` and ``csv`` must be configured. CSV paths are
    resolved relative to the source configuration file when ``base_directory``
    is supplied.

    :param configuration: ``experiment.input`` configuration mapping.
    :type configuration: Mapping[str, Any]
    :param base_directory: Directory used to resolve a relative CSV path.
    :type base_directory: pathlib.Path | None
    :return: Starting sequences, expanded by optional CSV repetitions.
    :rtype: list[str]
    :raises ValueError: If the input source or its values are invalid.
    """
    inline = configuration.get("sequences")
    csv_configuration = configuration.get("csv")
    if (inline is None) == (csv_configuration is None):
        raise ValueError("Configure exactly one of experiment.input.sequences or csv.")
    if inline is not None:
        return _validate_inline_sequences(inline)
    if not isinstance(csv_configuration, Mapping):
        raise ValueError("experiment.input.csv must be a mapping.")
    return _load_csv_sequences(csv_configuration, base_directory=base_directory)


def _validate_inline_sequences(value: Any) -> list[str]:
    """Validate and return an inline sequence list."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("experiment.input.sequences must be a sequence of strings.")
    sequences = list(value)
    if not sequences or not all(isinstance(sequence, str) and sequence for sequence in sequences):
        raise ValueError("Input sequences must contain non-empty strings.")
    return sequences


def _load_csv_sequences(
    configuration: Mapping[str, Any],
    *,
    base_directory: Path | None,
) -> list[str]:
    """Read and expand peptide sequences from a configured CSV file."""
    raw_path = configuration.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("experiment.input.csv.path must be a non-empty string.")
    path = Path(raw_path)
    if not path.is_absolute() and base_directory is not None:
        path = base_directory / path
    sequence_column = configuration.get("sequence_column", "sequence")
    repetitions_column = configuration.get("repetitions_column", "repetitions")
    sequences: list[str] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or sequence_column not in reader.fieldnames:
            raise ValueError(
                f"Input CSV must contain the configured sequence column: {sequence_column}"
            )
        for row_number, row in enumerate(reader, start=2):
            sequence = row.get(sequence_column, "").strip()
            if not sequence:
                raise ValueError(f"Input CSV row {row_number} has an empty sequence.")
            raw_repetitions = row.get(repetitions_column, "1") or "1"
            try:
                repetitions = int(raw_repetitions)
            except ValueError as error:
                raise ValueError(
                    f"Input CSV row {row_number} has invalid repetitions: {raw_repetitions}"
                ) from error
            if repetitions < 1:
                raise ValueError(
                    f"Input CSV row {row_number} repetitions must be positive."
                )
            sequences.extend([sequence] * repetitions)
    if not sequences:
        raise ValueError("Input CSV must contain at least one sequence.")
    return sequences
