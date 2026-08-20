"""Consistent visualizations for locality analysis results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

from pep_compass.analysis.result import AnalysisResult
from pep_compass.analysis.visualization.theme import PlotTheme


class LocalityVisualizer:
    """Render locality results using one shared theme."""

    def __init__(self, theme: PlotTheme) -> None:
        self.theme = theme

    @staticmethod
    def share_y_limits(figures: dict) -> None:
        """Apply one common y-range across every ``Axes`` in a figures dict.

        Plotting methods that return multiple independent figures (e.g.
        :meth:`latent_jump`, :meth:`latent_jump_by_iteration`) deliberately
        don't equalize axes themselves -- call this on the returned dict
        before displaying when the figures need to be visually comparable.

        :param figures: A ``{key: Axes}`` mapping, e.g. one method's return value.
        """
        axes_list = list(figures.values())
        if not axes_list:
            return
        bottom = min(axes.get_ylim()[0] for axes in axes_list)
        top = max(axes.get_ylim()[1] for axes in axes_list)
        for axes in axes_list:
            axes.set_ylim(bottom, top)

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

    def latent_jump(self, result: AnalysisResult) -> dict[tuple[bool, object], "plt.Axes"]:
        """Plot Euclidean latent distance vs. Levenshtein distance, one figure
        per ``(sorbes_moved, grid_value)`` combination.

        Separate figures rather than facets/hues for two reasons: comparing
        "mutang acceptance" (grid) settings as whole plots was the explicit
        requirement, and pooling seeds whose SORBES walker never moved with
        seeds where it did would average away exactly the effect this plot is
        meant to show (see ``sorbes_moved`` in :func:`latent_jump`'s docstring).
        Keyed and iterated movement-first so stationary-seed and moving-seed
        figures are each contiguous rather than interleaved by grid value.

        :param result: Output of ``LocalityAnalysis.latent_jump``.
        :return: Mapping from ``(sorbes_moved, grid_value)`` to its ``Axes``.
        """
        data = result.data
        if data.empty:
            raise ValueError("Cannot visualize an empty latent-jump result")
        labels = {
            "candidate_to_origin": "candidate → trajectory origin",
            "candidate_to_sorbes_parent": "candidate → generating SORBES point",
            "sorbes_to_origin": "SORBES point → trajectory origin",
        }
        figures = {}
        # Grouped by movement first so stationary-seed figures and
        # moving-seed figures are each contiguous when iterated/displayed,
        # rather than interleaving by grid value.
        for (sorbes_moved, grid_value), panel in data.groupby(
            ["sorbes_moved", "grid_value"], dropna=False, sort=True
        ):
            with sns.axes_style(self.theme.style), sns.plotting_context(
                self.theme.context
            ):
                figure, axes = plt.subplots(
                    figsize=(
                        self.theme.figure_size[0] * 1.6,
                        self.theme.figure_size[1],
                    ),
                    dpi=self.theme.dpi,
                )
            plot_data = panel.copy()
            plot_data["distance_kind"] = plot_data["distance_kind"].map(labels)
            sns.violinplot(
                data=plot_data,
                x="levenshtein_distance",
                y="euclidean_distance",
                hue="distance_kind",
                palette=self.theme.palette,
                cut=0,
                density_norm="width",
                ax=axes,
            )
            movement_label = "moving" if sorbes_moved else "stationary"
            axes.set(
                title=(
                    "Euclidean latent distance vs. Levenshtein edit distance\n"
                    f"direction_significance_threshold={grid_value:g}, "
                    f"SORBES {movement_label} seeds"
                ),
                xlabel="Levenshtein distance",
                ylabel="Euclidean latent distance",
            )
            axes.legend(title="", loc="upper left", fontsize="small")
            figure.tight_layout()
            figures[(sorbes_moved, grid_value)] = axes
        return figures

    def latent_jump_by_iteration(
        self, result: AnalysisResult
    ) -> dict[tuple[bool, object], "plt.Axes"]:
        """Plot Euclidean latent distance vs. local-enumeration iteration.

        Same three distance kinds and same ``(sorbes_moved, grid_value)``
        figure split as :meth:`latent_jump`, but x is ``trajectory_step``
        (the generating SORBES iteration) instead of Levenshtein distance --
        i.e. how each distance's *distribution* evolves step by step, not how
        it relates to edit distance. Rows with no resolved ``trajectory_step``
        (unmatched SORBES parent) are dropped.

        :param result: Output of ``LocalityAnalysis.latent_jump``.
        :return: Mapping from ``(sorbes_moved, grid_value)`` to its ``Axes``.
        """
        data = result.data.dropna(subset=["trajectory_step"])
        if data.empty:
            raise ValueError("Cannot visualize an empty latent-jump result")
        labels = {
            "candidate_to_origin": "candidate → trajectory origin",
            "candidate_to_sorbes_parent": "candidate → generating SORBES point",
            "sorbes_to_origin": "SORBES point → trajectory origin",
        }
        figures = {}
        for (sorbes_moved, grid_value), panel in data.groupby(
            ["sorbes_moved", "grid_value"], dropna=False, sort=True
        ):
            with sns.axes_style(self.theme.style), sns.plotting_context(
                self.theme.context
            ):
                figure, axes = plt.subplots(
                    figsize=(
                        self.theme.figure_size[0] * 1.6,
                        self.theme.figure_size[1],
                    ),
                    dpi=self.theme.dpi,
                )
            plot_data = panel.copy()
            plot_data["trajectory_step"] = plot_data["trajectory_step"].astype(int)
            plot_data["distance_kind"] = plot_data["distance_kind"].map(labels)
            sns.violinplot(
                data=plot_data,
                x="trajectory_step",
                y="euclidean_distance",
                hue="distance_kind",
                palette=self.theme.palette,
                cut=0,
                density_norm="width",
                ax=axes,
            )
            movement_label = "moving" if sorbes_moved else "stationary"
            axes.set(
                title=(
                    "Euclidean latent distance vs. local-enumeration iteration\n"
                    f"direction_significance_threshold={grid_value:g}, "
                    f"SORBES {movement_label} seeds"
                ),
                xlabel="Local-enumeration iteration",
                ylabel="Euclidean latent distance",
            )
            axes.legend(title="", loc="upper left", fontsize="small")
            figure.tight_layout()
            figures[(sorbes_moved, grid_value)] = axes
        return figures

    def sorbes_trajectory_profile(self, result: AnalysisResult):
        """Plot each trajectory's latent drift from its origin, by iteration.

        One thin line per trajectory (``units="trajectory_id"``, unaggregated),
        colored by seed sequence, so stationary and moving seeds are visible
        as flat-vs-rising line bundles at a glance.

        :param result: Output of ``LocalityAnalysis.sorbes_trajectory_profile``.
        """
        data = result.data
        if data.empty:
            raise ValueError("Cannot visualize an empty SORBES trajectory profile")
        _, axes = self._axes()
        sns.lineplot(
            data=data,
            x="trajectory_step",
            y="euclidean_distance",
            hue="seed_sequence",
            units="trajectory_id",
            estimator=None,
            alpha=0.6,
            palette=self.theme.palette,
            ax=axes,
        )
        axes.set(
            title="SORBES latent drift from trajectory origin",
            xlabel="Local-enumeration iteration",
            ylabel="Euclidean latent distance",
        )
        axes.legend(title="seed", loc="upper left", fontsize="small")
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

    def _method_comparison_grid(
        self,
        result: AnalysisResult,
        y: str,
        title: str,
        ylabel: str,
    ):
        """Plot peptide-level method outcomes in separate locality panels."""
        data = result.data.copy()
        if data.empty:
            raise ValueError("Cannot visualize an empty method comparison")
        radii = sorted(data["levenshtein_radius"].dropna().unique())[:8]
        methods = sorted(data["method"].dropna().astype(str).unique())
        colors = sns.color_palette(self.theme.palette, n_colors=len(methods))
        palette = dict(zip(methods, colors))
        with sns.axes_style(self.theme.style), sns.plotting_context(
            self.theme.context
        ):
            figure, axes = plt.subplots(
                2,
                4,
                figsize=(
                    self.theme.figure_size[0] * 2.4,
                    self.theme.figure_size[1] * 1.8,
                ),
                dpi=self.theme.dpi,
                sharey=True,
                constrained_layout=True,
            )
        for axes_item, radius in zip(axes.flat, radii):
            panel = data[data["levenshtein_radius"] == radius]
            per_peptide = (
                panel.groupby(
                    ["method", "method_variant", "name"], dropna=False
                )[y]
                .median()
                .reset_index()
            )
            order = per_peptide["method_variant"].drop_duplicates().tolist()
            sns.stripplot(
                data=per_peptide,
                x="method_variant",
                y=y,
                hue="method",
                order=order,
                palette=palette,
                jitter=0.18,
                size=5,
                alpha=0.8,
                ax=axes_item,
            )
            medians = per_peptide.groupby("method_variant", dropna=False)[y].median()
            for position, variant in enumerate(order):
                axes_item.scatter(
                    position,
                    medians.loc[variant],
                    marker="_",
                    s=180,
                    linewidth=2.5,
                    color="black",
                    zorder=5,
                )
            if axes_item.legend_ is not None:
                axes_item.legend_.remove()
            axes_item.set_title(f"Levenshtein ≤ {int(radius)}")
            axes_item.set_xlabel("")
            axes_item.set_ylabel(ylabel)
            axes_item.tick_params(axis="x", rotation=55)
        for axes_item in axes.flat[len(radii):]:
            axes_item.set_visible(False)
        handles = [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                color=palette[method],
                label=method,
            )
            for method in methods
        ]
        figure.legend(
            handles=handles,
            loc="outside upper center",
            ncol=min(5, len(handles)),
        )
        figure.suptitle(title)
        return axes

    def method_comparison(self, result: AnalysisResult):
        """Plot local candidate yield for every method and random control."""
        return self._method_comparison_grid(
            result,
            "post_method_local_unique",
            "Method comparison: retained local candidates",
            "Unique candidates after method filter",
        )

    def method_retention(self, result: AnalysisResult):
        """Plot local pool retention for every method and random control."""
        return self._method_comparison_grid(
            result,
            "method_retention_local",
            "Method comparison: local pool retention",
            "Post-method / pre-method unique candidates",
        )

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
