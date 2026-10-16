"""Direct peptide-length constraint."""

import torch

from pep_compass.steps.base import Step


class SequenceLength(Step):
    """Keep sequences whose lengths lie in an inclusive interval."""

    category = "filter"

    def __init__(self, minimum: int = 1, maximum: int | None = None) -> None:
        self._configure({"minimum": minimum, "maximum": maximum})
        if minimum < 0 or (maximum is not None and maximum < minimum):
            raise ValueError("Invalid sequence-length interval.")
        self.minimum, self.maximum = minimum, maximum

    def _execute(self, batch, context):
        del context
        retained = [
            index
            for index, sequence in enumerate(batch.sequences)
            if len(sequence) >= self.minimum
            and (self.maximum is None or len(sequence) <= self.maximum)
        ]
        return batch.select(torch.tensor(retained, device=batch.latent_origins.device))
