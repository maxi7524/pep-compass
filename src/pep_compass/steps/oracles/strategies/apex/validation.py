"""Pure APEX parameter constraints without loading predictors."""

from collections.abc import Sequence


def validate_bacteria_indices(
    value: str | Sequence[int],
    *,
    pathogen_count: int,
) -> tuple[int, ...] | None:
    """Normalize and validate configured pathogen columns."""
    if value == "all":
        return None
    if isinstance(value, str) or not value:
        raise ValueError("APEX mic_bacteria must be 'all' or a non-empty index list.")
    indices = tuple(value)
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < pathogen_count
        for index in indices
    ):
        raise ValueError("APEX mic_bacteria contains an invalid pathogen index.")
    return indices
