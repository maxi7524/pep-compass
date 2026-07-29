"""Consistent visualizations for locality analysis results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

from pep_compass.experiments.analysis.result import AnalysisResult
from pep_compass.experiments.analysis.visualization.theme import PlotTheme


class LocalityVisualizer:
    """Render locality results using one shared theme."""

    def __init__(self, theme: PlotTheme) -> None:
        self.theme = theme

    def _axes(self):
        with sns.axes_style(self.theme.style), sns.plotting_context(self.theme.context):
            figure, axes = plt.subplots(
                figsize=self.theme.figure_size, dpi=self.theme.dpi
            )
        return figure, axes

    def _hue(self, data, column: str) -> dict:
        if column in data and data[column].notna().any():
            return {"hue": column, "palette": self.theme.palette}
        return {}

    def trajectory_saturation(self, result: AnalysisResult):
        """Plot unique candidate yield against sampled trajectory count."""
        _, axes = self._axes()
        data = result.data
        sns.lineplot(
            data=data,
            x="trajectory_count",
            y="unique_candidates_mean",
            **self._hue(data, "mutation.token_threshold"),
            style="method",
            marker="o",
            ax=axes,
        )
        axes.set(title="Trajectory saturation", ylabel="Unique candidates")
        return axes

    def step_saturation(self, result: AnalysisResult):
        """Plot unique candidate yield against maximum walker step."""
        _, axes = self._axes()
        sns.lineplot(
            data=result.data,
            x="max_step",
            y="unique_candidates_mean",
            **self._hue(result.data, "mutation.token_threshold"),
            style="method",
            marker="o",
            ax=axes,
        )
        axes.set(title="Step saturation", ylabel="Unique candidates")
        return axes

    def latent_locality(self, result: AnalysisResult):
        """Plot latent distance distribution summaries by edit distance."""
        _, axes = self._axes()
        sns.lineplot(
            data=result.data,
            x="levenshtein_to_center",
            y="latent_distance_median",
            marker="o",
            color=sns.color_palette(self.theme.palette)[0],
            ax=axes,
        )
        axes.fill_between(
            result.data["levenshtein_to_center"],
            result.data["latent_distance_q25"],
            result.data["latent_distance_q75"],
            alpha=self.theme.confidence_alpha,
        )
        axes.set(title="Latent locality", ylabel="Walker-to-candidate distance")
        return axes

    def mutang_selectivity(self, result: AnalysisResult):
        """Plot retained unique candidates against the MUTANG threshold."""
        _, axes = self._axes()
        sns.lineplot(
            data=result.data,
            x="mutation.token_threshold",
            y="post_constraint_unique_mean",
            **self._hue(result.data, "method"),
            marker="o",
            ax=axes,
        )
        axes.set_xscale("log")
        axes.set(title="MUTANG selectivity", ylabel="Retained unique candidates")
        return axes

    def parameter_selection(self, result: AnalysisResult):
        """Plot candidate yield against constraint retention."""
        _, axes = self._axes()
        sns.scatterplot(
            data=result.data,
            x="constraint_retention_mean",
            y="candidates_per_trajectory_mean",
            **self._hue(result.data, "mutation.token_threshold"),
            style="method",
            s=80,
            ax=axes,
        )
        axes.set(
            title="Preliminary parameter selection",
            xlabel="Constraint retention",
            ylabel="Unique candidates per trajectory",
        )
        return axes
