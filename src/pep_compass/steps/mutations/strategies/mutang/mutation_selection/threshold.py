"""Baseline threshold selection of single mutations."""

from pep_compass.steps.mutations.strategies.mutang.base import (
    MutationSelectionStrategy,
)
from pep_compass.steps.mutations.strategies.mutang.schema import (
    MutationOption,
)


class ThresholdSelection(MutationSelectionStrategy):
    """Select every non-padding residue above the token threshold."""

    def __init__(self, threshold=0.1):
        from pep_compass.utils.validation import validate_positive

        validate_positive(threshold, "ThresholdSelection.threshold", allow_zero=True)
        self._configure({"threshold": threshold})
        self.threshold = threshold

    def select(self, scores, sequence_length):
        rows = (scores[:sequence_length, 1:] > self.threshold).nonzero(as_tuple=False)
        return tuple(
            MutationOption(
                int(position), int(residue) + 1, float(scores[position, residue + 1])
            )
            for position, residue in rows.cpu().tolist()
        )
