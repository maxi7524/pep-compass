"""Base contracts for filters driven by reusable candidate scores."""

from pep_compass.steps.base import Step
from pep_compass.steps.filters.strategies.ranked.scoring.base import ScoreFunction
from pep_compass.steps.filters.strategies.ranked.selection.base import SelectionRule


class Ranked(Step):
    """Compose one score function with one score-based selection rule."""

    stage_parameters = ("scoring", "selection")
    category = "filter"

    def __init__(self, scoring: ScoreFunction, selection: SelectionRule) -> None:
        self._configure({"scoring": scoring, "selection": selection})
        self.scoring, self.selection = scoring, selection

    required_resources = ("autoencoder",)

    def contract(self):
        from pep_compass.data_schemas.contracts import StepContract

        contract = getattr(self.scoring, "contract", None)
        return (
            contract()
            if contract is not None
            else StepContract(resources=self.required_resources)
        )

    def _execute(self, batch, context):
        scores = self.scoring(batch, context)  # (B,)
        if scores.ndim != 1 or scores.shape[0] != len(batch):
            raise ValueError(
                "Ranked filter scoring must return one score per candidate."
            )
        indices = self.selection.select(
            scores, higher_is_better=self.scoring.higher_is_better, context=context
        )
        return batch.select(indices)
