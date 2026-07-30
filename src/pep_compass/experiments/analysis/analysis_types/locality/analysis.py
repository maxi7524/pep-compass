"""Public facade collecting locality experiment analyses by research question."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pep_compass.experiments.analysis.analysis_types.locality.latent_geometry import (
    latent_locality,
)
from pep_compass.experiments.analysis.analysis_types.locality.mutang import (
    mutang_selectivity,
)
from pep_compass.experiments.analysis.analysis_types.locality.parameters import (
    parameter_selection,
)
from pep_compass.experiments.analysis.analysis_types.locality.sorbes import (
    step_saturation,
    trajectory_saturation,
)
from pep_compass.experiments.analysis.result import AnalysisResult
from pep_compass.experiments.reader.selection import ExperimentSelection


class LocalityAnalysis:
    """Expose SORBES, latent, MUTANG, and parameter locality analyses."""

    def __init__(self, selection: ExperimentSelection, use_cache: bool = True) -> None:
        self.selection = selection
        self.use_cache = use_cache

    def _run(
        self,
        name: str,
        operation: Callable[..., AnalysisResult],
        parameters: dict[str, Any],
    ) -> AnalysisResult:
        store = self.selection.reader.metrics
        specification = self.selection.specification()
        analysis_version = "4"
        if self.use_cache:
            cached = store.get_analysis(
                name, specification, parameters, analysis_version
            )
            if cached is not None:
                frame, metadata = cached
                return AnalysisResult(frame, metadata, {"cache_hit": True})
        result = operation(self.selection, **parameters)
        store.put_analysis(
            name,
            result.data,
            result.metadata,
            specification,
            parameters,
            analysis_version,
        )
        return result

    def trajectory_saturation(self, **parameters: Any) -> AnalysisResult:
        """Estimate diminishing candidate yield as trajectories are added."""
        return self._run(
            "locality.trajectory_saturation",
            trajectory_saturation,
            parameters,
        )

    def step_saturation(self, **parameters: Any) -> AnalysisResult:
        """Estimate diminishing candidate yield as walker depth increases."""
        return self._run("locality.step_saturation", step_saturation, parameters)

    def latent_locality(self, **parameters: Any) -> AnalysisResult:
        """Relate edit distance to actual and encoded latent positions."""
        return self._run("locality.latent_locality", latent_locality, parameters)

    def mutang_selectivity(self, **parameters: Any) -> AnalysisResult:
        """Summarize threshold, dimensionality, and candidate-retention effects."""
        return self._run(
            "locality.mutang_selectivity", mutang_selectivity, parameters
        )

    def parameter_selection(self, **parameters: Any) -> AnalysisResult:
        """Calculate preliminary multi-objective parameter summaries."""
        return self._run(
            "locality.parameter_selection", parameter_selection, parameters
        )
