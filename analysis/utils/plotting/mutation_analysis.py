import numpy as np
import matplotlib.pyplot as plt


def plot_mean_vs_std_bubble(
    df,
    mean_col="mean",
    std_col="std",
    n_col="n",
    title="Mutation Statistics: Mean vs Standard Deviation",
    figsize=(10, 8),
    alpha=0.75,
    edgecolor="k",
    linewidth=0.35,
    size_range=(12, 220),  # (min_marker_area, max_marker_area) in points^2
    size_scale="sqrt",  # "sqrt" or "linear"
    legend_ns=None,  # e.g. [26, 116, 291]
    legend_loc="upper right",
    grid=True,
    ax=None,
    label=None,  # optional: show population label in legend
    color=None,  # optional: fix color for a population
):
    # checks
    for c in (mean_col, std_col, n_col):
        if c not in df.columns:
            raise KeyError(f"Missing column '{c}'. Available: {list(df.columns)}")

    x = df[mean_col].to_numpy(dtype=float)
    y = df[std_col].to_numpy(dtype=float)
    n = df[n_col].to_numpy(dtype=float)

    # Clean finite values (FIXED)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(n) & (n > 0)
    x, y, n = x[ok], y[ok], n[ok]

    # size mapping
    smin, smax = size_range
    n_min, n_max = float(np.nanmin(n)), float(np.nanmax(n))

    if n_min == n_max:
        sizes = np.full_like(n, (smin + smax) / 2.0, dtype=float)
    else:
        if size_scale == "sqrt":
            nn = np.sqrt(n)
            a, b = np.sqrt(n_min), np.sqrt(n_max)
        elif size_scale == "linear":
            nn = n
            a, b = n_min, n_max
        else:
            raise ValueError("size_scale must be 'sqrt' or 'linear'")

        t = (nn - a) / (b - a)
        sizes = smin + t * (smax - smin)

    # axes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    if grid:
        ax.grid(True, alpha=0.28)

    # scatter
    ax.scatter(
        x,
        y,
        s=sizes,
        alpha=alpha,
        edgecolors=edgecolor,
        linewidths=linewidth,
        color=color,  # None -> matplotlib default cycle
        label=label,  # None -> no population legend entry
    )

    # labels (only set title/axes once, ideally from the first call)
    ax.set_xlabel("Mean")
    ax.set_ylabel("Std")

    # bubble legend (sample size) — only add if requested
    if legend_ns is not None:

        def n_to_size(nv):
            nv = float(nv)
            if n_min == n_max:
                return (smin + smax) / 2.0
            if size_scale == "sqrt":
                nnv = np.sqrt(nv)
                a2, b2 = np.sqrt(n_min), np.sqrt(n_max)
            else:
                nnv = nv
                a2, b2 = n_min, n_max
            tt = (nnv - a2) / (b2 - a2)
            return smin + tt * (smax - smin)

        handles = [
            ax.scatter(
                [],
                [],
                s=n_to_size(v),
                alpha=alpha,
                edgecolors=edgecolor,
                linewidths=linewidth,
                color="C0",
            )
            for v in legend_ns
        ]
        labels_ = [f"n={int(v)}" for v in legend_ns]

        ax.legend(handles, labels_, title="Sample size", loc=legend_loc, frameon=True)

    if title is not None:
        ax.set_title(title, fontweight="bold")

    return ax


def aa_mutation_heatmap(
    df_mean: pd.DataFrame,
    aa_list: list[str],
    df_std: pd.DataFrame | None = None,
    df_n: pd.DataFrame | None = None,
    AA_group: dict[str, list[str]] | None = None,
    cmap: str = "magma",
    nan_color: str = "black",
    figsize=(9, 7),
    fontsize: int = 11,
    title_fontsize: int = 12,
    label_fontsize: int = 11,
    title: str = "",
    x_label: str = "To AA n-gram",
    y_label: str = "From AA n-gram",
    group_linewidth: float = 1.5,
    group_colors: dict[str, str] | None = None,
    # NEW: separate axis control
    x_bar_gap: float = 0.0,  # heatmap → X-axis color bar
    x_label_gap: float = 0.0,  # color bar → X-axis label
    y_bar_gap: float = 0.0,  # heatmap → Y-axis color bar
    y_label_gap: float = 0.0,  # color bar → Y-axis label
    group_label_color: str = "black",
    # NEW: colorbar control
    vmin: float | None = None,  # minimum value for colorbar
    vmax: float | None = None,  # maximum value for colorbar
    cbar_shrink: float = 0.8,  # shrink factor for colorbar (0-1)
    cbar_aspect: float = 20,  # aspect ratio of colorbar
    cbar_pad: float = 0.02,  # padding between heatmap and colorbar
    # NEW: threshold highlighting
    df_threshold: (
        pd.DataFrame | None
    ) = None,  # dataframe with values to compare against threshold
    threshold: float | None = None,  # threshold value
    threshold_direction: str = "both",  # "above", "below", or "both"
    threshold_color: str = "red",  # color of threshold rectangle
    threshold_linewidth: float = 2.0,  # linewidth of threshold rectangle
):
    """
    Heatmap with fully symmetric and independently controlled
    X/Y axis group bars and group labels.
    """

    df = df_mean.copy()

    # === 1. Group sorting ===
    if AA_group is not None:
        sorted_aa = []
        group_sizes, group_names = [], []

        for gname, members in AA_group.items():
            present = [aa for aa in aa_list if aa in members]
            if present:
                sorted_aa.extend(present)
                group_sizes.append(len(present))
                group_names.append(gname)

        df = df.loc[sorted_aa, sorted_aa]
        if df_std is not None:
            df_std = df_std.loc[sorted_aa, sorted_aa]
        if df_n is not None:
            df_n = df_n.loc[sorted_aa, sorted_aa]
        if df_threshold is not None:
            df_threshold = df_threshold.loc[sorted_aa, sorted_aa]
    else:
        sorted_aa = aa_list
        group_sizes, group_names = [], []
        if df_threshold is not None:
            df_threshold = df_threshold.loc[sorted_aa, sorted_aa]

    # === 2. Annotation matrix ===
    annot = np.empty(df.shape, dtype=object)
    for i, r in enumerate(df.index):
        for j, c in enumerate(df.columns):
            if pd.isna(df.loc[r, c]):
                annot[i, j] = ""
                continue
            txt = f"{df.loc[r, c]:.2f}"
            if df_std is not None:
                sd = df_std.loc[r, c]
                if not pd.isna(sd):
                    txt += f"\n± {sd:.2f}"
            if df_n is not None:
                n = df_n.loc[r, c]
                if not pd.isna(n):
                    txt += f"\n(n={int(n)})"
            annot[i, j] = txt

    # === 3. NaN color ===
    base_cmap = plt.get_cmap(cmap).copy()
    base_cmap.set_bad(color=nan_color)
    mask = pd.isna(df)

    # === 4. Draw heatmap ===
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        df,
        cmap=base_cmap,
        annot=annot,
        fmt="",
        mask=mask,
        square=True,
        cbar=True,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={
            "shrink": cbar_shrink,
            "aspect": cbar_aspect,
            "pad": cbar_pad,
        },
        annot_kws={"fontsize": fontsize - 1},
        ax=ax,
    )

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(x_label, fontsize=label_fontsize)
    ax.set_ylabel(y_label, fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=label_fontsize)

    # === 4.5. Threshold highlighting ===
    if df_threshold is not None and threshold is not None:
        # Determine which cells to highlight
        highlight_mask = pd.DataFrame(
            False, index=df_threshold.index, columns=df_threshold.columns
        )

        if threshold_direction == "above":
            highlight_mask = df_threshold > threshold
        elif threshold_direction == "below":
            highlight_mask = df_threshold < threshold
        elif threshold_direction == "both":
            highlight_mask = (df_threshold > threshold) | (df_threshold < threshold)
        else:
            raise ValueError(
                f"threshold_direction must be 'above', 'below', or 'both', got '{threshold_direction}'"
            )

        # Draw rectangles around highlighted cells
        # Note: heatmap cells are centered at integer positions (0.5, 1.5, 2.5, ...)
        for i, row_idx in enumerate(df_threshold.index):
            for j, col_idx in enumerate(df_threshold.columns):
                if highlight_mask.loc[row_idx, col_idx] and not pd.isna(
                    df_threshold.loc[row_idx, col_idx]
                ):
                    # Draw rectangle around the cell
                    # Cell boundaries: [i, i+1] x [j, j+1] in data coordinates
                    rect = plt.Rectangle(
                        (j, i),  # bottom-left corner
                        1,  # width
                        1,  # height
                        fill=False,
                        edgecolor=threshold_color,
                        linewidth=threshold_linewidth,
                        transform=ax.transData,
                    )
                    ax.add_patch(rect)

    # === 5. Group boundaries + bars + labels (AXES COORDINATES) ===
    if AA_group is not None and group_sizes:

        boundaries = np.cumsum(group_sizes)
        mids = boundaries - np.array(group_sizes) / 2
        total = len(sorted_aa)

        # Convert mid positions from data coords → axes coords
        mids_axes = mids / total
        size_axes = np.array(group_sizes) / total

        bar_thickness_axes = 0.02  # thickness as fraction of axis

        # Separator lines (still in data coordinates)
        for b in boundaries[:-1]:
            ax.axhline(b, color="black", lw=group_linewidth)
            ax.axvline(b, color="black", lw=group_linewidth)

        # Draw bars + labels in AXES COORDINATES
        for mid_ax, gname, gsize_ax in zip(mids_axes, group_names, size_axes):

            col = group_colors.get(gname, "lightgray") if group_colors else "lightgray"

            # === Y-axis color bar ===
            # Reverse Y-axis position to match heatmap order (top to bottom in data = bottom to top in axes)
            mid_ax_y_reversed = 1.0 - mid_ax
            ax.add_patch(
                plt.Rectangle(
                    (-y_bar_gap, mid_ax_y_reversed - gsize_ax / 2),
                    bar_thickness_axes,
                    gsize_ax,
                    transform=ax.transAxes,
                    clip_on=False,
                    color=col,
                )
            )

            # === Y-axis label ===
            ax.text(
                -y_bar_gap - bar_thickness_axes - y_label_gap,
                mid_ax_y_reversed,
                gname,
                ha="center",
                va="center",
                transform=ax.transAxes,
                rotation=90,
                fontsize=fontsize,
                color=group_label_color,
            )

            # === X-axis color bar (BOTTOM) ===
            ax.add_patch(
                plt.Rectangle(
                    (mid_ax - gsize_ax / 2, -x_bar_gap),
                    gsize_ax,
                    bar_thickness_axes,
                    transform=ax.transAxes,
                    clip_on=False,
                    color=col,
                )
            )

            # === X-axis label (BOTTOM) ===
            ax.text(
                mid_ax,
                -x_bar_gap - bar_thickness_axes - x_label_gap,
                gname,
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=fontsize,
                color=group_label_color,
            )

    fig.tight_layout()
    return fig, ax


from scipy.stats import mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import multipletests
import numpy as np
import pandas as pd
from typing import Callable


def mutation_mannwhitney_test(
    dict_a: dict,
    dict_b: dict | None = None,  # NEW: optional, if None test against zero
    value_key: str = None,
    subset: list[str] = None,  # REQUIRED
    alternative: str = "two-sided",
    agg_func: Callable = np.mean,  # NEW: aggregation function
    agg_func_kwargs: dict | None = None,  # NEW: kwargs for aggregation function
    # NEW: kwargs
    mannwhitney_kwargs: dict | None = None,  # passed into mannwhitneyu
    wilcoxon_kwargs: dict | None = None,  # passed into wilcoxon (when dict_b is None)
    multitest: bool = False,  # whether to apply BH correction
    multitest_kwargs: dict | None = None,  # kwargs passed into multipletests
):
    """
    Performs Mann–Whitney U test for each mutation (aa1 -> aa2)
    comparing values from dict_a vs dict_b.

    If dict_b is None, performs one-sample Wilcoxon signed-rank test
    against zero for values from dict_a.

    Supports optional multiple testing correction (Benjamini–Hochberg).

    Parameters
    ----------
    dict_a : dict
        Mutation dictionary.
    dict_b : dict or None
        Mutation dictionary. If None, tests dict_a values against zero.
    value_key : str
        Inner key to extract values from.
    subset : list[str]
        REQUIRED list of variable names to include.
    alternative : str
        two-sided, less, greater (Mann–Whitney argument)
    agg_func : callable, default np.mean
        Aggregation function to apply over the subset variables (e.g., np.mean, np.max,
        np.min, lambda x: np.quantile(x, 0.95), etc.).
    agg_func_kwargs : dict, optional
        Additional keyword arguments to pass to the aggregation function.
    mannwhitney_kwargs : dict
        Additional arguments forwarded to scipy.stats.mannwhitneyu.
    wilcoxon_kwargs : dict
        Additional arguments forwarded to scipy.stats.wilcoxon (when dict_b is None).
    multitest : bool
        Whether to apply multiple testing correction (Benjamini–Hochberg).
    multitest_kwargs : dict
        Extra keyword arguments passed to statsmodels.multipletests().

    Returns
    -------
    df_p_raw : pivot table of raw p-values
    df_p_adj : pivot table of corrected p-values (NaN if multitest=False)
    df_u      : pivot table of U statistics (or W statistics if one-sample)
    df_n1     : pivot table of sample counts in dict_a
    df_n2     : pivot table of sample counts in dict_b (NaN if one-sample)
    df_effect : pivot table of median difference (median(a) - median(b)) or median(a) if one-sample
    """

    if mannwhitney_kwargs is None:
        mannwhitney_kwargs = {}
    if wilcoxon_kwargs is None:
        wilcoxon_kwargs = {}
    if multitest_kwargs is None:
        multitest_kwargs = {}
    if agg_func_kwargs is None:
        agg_func_kwargs = {}

    rows = []
    one_sample = dict_b is None

    # union of all mutation pairs
    if one_sample:
        all_keys = set(dict_a.keys())
    else:
        all_keys = set(dict_a.keys()).union(dict_b.keys())

    for aa1, aa2 in all_keys:

        def extract_values(rec):
            if rec is None or value_key not in rec:
                return []
            all_values = []
            for var, vals in rec[value_key].items():
                if var in subset:
                    all_values.append(vals)

            if len(all_values) == 0:
                return []

            arr = np.array(all_values, dtype=float)
            agg_arr = agg_func(arr, **agg_func_kwargs)
            assert (
                agg_arr.ndim == 1
            ), f"Aggregation function {agg_func} returned array with {agg_arr.ndim} dimensions"
            return list(agg_arr)

        vals_a = extract_values(dict_a.get((aa1, aa2)))

        if one_sample:
            # One-sample test against zero
            if len(vals_a) == 0:
                w_stat = np.nan
                p_val = np.nan
                effect = np.nan
            else:
                try:
                    w_stat, p_val = wilcoxon(
                        vals_a, alternative=alternative, **wilcoxon_kwargs
                    )
                    effect = np.median(vals_a)  # median value (difference from zero)
                except Exception:
                    w_stat = np.nan
                    p_val = np.nan
                    effect = np.nan

            rows.append(
                {
                    "from_aa": aa1,
                    "to_aa": aa2,
                    "p_raw": p_val,
                    "u": w_stat,  # storing W statistic in 'u' column for consistency
                    "n1": len(vals_a),
                    "n2": np.nan,
                    "effect": effect,
                }
            )
        else:
            # Two-sample test
            vals_b = extract_values(dict_b.get((aa1, aa2)))

            # if one side has zero samples → cannot test
            if len(vals_a) == 0 or len(vals_b) == 0:
                u_stat = np.nan
                p_val = np.nan
                effect = np.nan
            else:
                try:
                    u_stat, p_val = mannwhitneyu(
                        vals_a, vals_b, alternative=alternative, **mannwhitney_kwargs
                    )
                    effect = np.median(vals_a) - np.median(vals_b)

                except Exception:
                    u_stat = np.nan
                    p_val = np.nan
                    effect = np.nan

            rows.append(
                {
                    "from_aa": aa1,
                    "to_aa": aa2,
                    "p_raw": p_val,
                    "u": u_stat,
                    "n1": len(vals_a),
                    "n2": len(vals_b),
                    "effect": effect,
                }
            )

    df = pd.DataFrame(rows)

    # --- Multiple testing correction ---
    if multitest:
        pvals = df["p_raw"].values
        _, p_adj, _, _ = multipletests(
            pvals, method="fdr_bh", **multitest_kwargs  # Benjamini-Hochberg
        )
        df["p_adj"] = p_adj
    else:
        df["p_adj"] = np.nan

    # --- Make pivot tables ---
    df_p_raw = df.pivot(index="from_aa", columns="to_aa", values="p_raw")
    df_p_adj = df.pivot(index="from_aa", columns="to_aa", values="p_adj")
    df_u = df.pivot(index="from_aa", columns="to_aa", values="u")
    df_n1 = df.pivot(index="from_aa", columns="to_aa", values="n1")
    df_n2 = df.pivot(index="from_aa", columns="to_aa", values="n2")
    df_effect = df.pivot(index="from_aa", columns="to_aa", values="effect")

    return df_p_raw, df_p_adj, df_u, df_n1, df_n2, df_effect
