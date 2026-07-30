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

    def _saturation_grid(
        self,
        result: AnalysisResult,
        x: str,
        y: str,
        title: str,
        ylabel: str,
    ):
        """Plot per-peptide median and IQR in cumulative Levenshtein panels."""
        data = result.data.copy()
        if data.empty:
            raise ValueError("Cannot visualize an empty saturation result")
        radii = sorted(data["levenshtein_radius"].dropna().unique())[:8]
        with sns.axes_style(self.theme.style), sns.plotting_context(
            self.theme.context
        ):
            figure, axes = plt.subplots(
                2,
                4,
                figsize=(
                    self.theme.figure_size[0] * 2.0,
                    self.theme.figure_size[1] * 1.7,
                ),
                dpi=self.theme.dpi,
                sharex=True,
                sharey=True,
                constrained_layout=True,
            )
        color_values = sns.color_palette(self.theme.palette, n_colors=10)
        legend_handles = {}
        for axes_item, radius in zip(axes.flat, radii):
            panel = data[data["levenshtein_radius"] == radius]
            group_columns = ["method", "mutation.token_threshold"]
            for color_index, (key, group) in enumerate(
                panel.groupby(group_columns, dropna=False, sort=True)
            ):
                # Replicates first collapse within a peptide so that every
                # starting sequence has equal weight in the distribution.
                per_peptide = (
                    group.groupby(["name", x], dropna=False)[y]
                    .median()
                    .reset_index()
                )
                summary = per_peptide.groupby(x, dropna=False)[y].agg(
                    median="median",
                    q25=lambda values: values.quantile(0.25),
                    q75=lambda values: values.quantile(0.75),
                ).reset_index()
                method, threshold = key
                threshold_label = (
                    f"{threshold:g}"
                    if isinstance(threshold, (int, float))
                    else str(threshold)
                )
                label = f"{method}, threshold={threshold_label}"
                color = color_values[color_index % len(color_values)]
                line = axes_item.plot(
                    summary[x],
                    summary["median"],
                    marker="o",
                    markersize=3,
                    color=color,
                    label=label,
                )[0]
                axes_item.fill_between(
                    summary[x],
                    summary["q25"],
                    summary["q75"],
                    color=color,
                    alpha=self.theme.confidence_alpha,
                )
                legend_handles[label] = line
            axes_item.set_title(f"Levenshtein ≤ {int(radius)}")
            axes_item.set_xlabel(x.replace("_", " "))
            axes_item.set_ylabel(ylabel)
        for axes_item in axes.flat[len(radii):]:
            axes_item.set_visible(False)
        figure.suptitle(title)
        if legend_handles:
            figure.legend(
                legend_handles.values(),
                legend_handles.keys(),
                loc="outside upper center",
                ncol=min(3, len(legend_handles)),
            )
        return axes

    def trajectory_saturation(self, result: AnalysisResult):
        """Plot trajectory rarefaction by cumulative sequence-locality radius."""
        return self._saturation_grid(
            result,
            "trajectory_count",
            "unique_candidates",
            "SORBES trajectory saturation",
            "Unique candidates (median and IQR)",
        )

    def trajectory_marginal(self, result: AnalysisResult):
        """Plot marginal candidate gain from each additional trajectory."""
        return self._saturation_grid(
            result,
            "trajectory_count",
            "marginal_per_trajectory",
            "Marginal SORBES trajectory yield",
            "New candidates per added trajectory",
        )

    def step_saturation(self, result: AnalysisResult):
        """Plot exact cumulative depth curves by sequence-locality radius."""
        return self._saturation_grid(
            result,
            "max_step",
            "unique_candidates",
            "SORBES step saturation",
            "Unique candidates (median and IQR)",
        )

    def step_marginal(self, result: AnalysisResult):
        """Plot new local candidates produced per active trajectory and step."""
        return self._saturation_grid(
            result,
            "max_step",
            "new_candidates_per_active_trajectory",
            "Marginal SORBES step yield",
            "New candidates per active trajectory",
        )

    def latent_locality(self, result: AnalysisResult):
        """Plot latent distance distribution summaries by edit distance."""
        _, axes = self._axes()
        colors = sns.color_palette(self.theme.palette, n_colors=3)
        metrics = (
            ("walker_candidate_distance", "candidate to generating walker"),
            ("origin_candidate_distance", "candidate to trajectory origin"),
            ("walker_origin_distance", "walker drift from origin"),
        )
        for color, (metric, label) in zip(colors, metrics):
            axes.plot(
                result.data["levenshtein_to_center"],
                result.data[f"{metric}_median"],
                marker="o",
                color=color,
                label=label,
            )
            axes.fill_between(
                result.data["levenshtein_to_center"],
                result.data[f"{metric}_q25"],
                result.data[f"{metric}_q75"],
                color=color,
                alpha=self.theme.confidence_alpha,
            )
        axes.legend()
        axes.set(title="Latent locality", ylabel="Euclidean latent distance")
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
