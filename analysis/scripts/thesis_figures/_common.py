"""Shared helpers for regenerating the thesis Results-chapter figures as vector PDFs.

All figure scripts in this directory import from here so the styling matches the source
notebooks (count_mic_changes / karol_mutational_signature_analysis / final_wasserstein) and
every figure is written as a real vector PDF into the thesis ``figures/`` folder.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Colormap, LinearSegmentedColormap


# --- Vector-PDF output settings: embed real fonts as editable text, not outlines. -----
matplotlib.rcParams.update({
    "pdf.fonttype": 42,      # TrueType (Type-42) -> selectable/searchable text in the PDF
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.bbox": "tight",
    "savefig.dpi": 300,      # only matters for any raster insets; lines/text stay vector
    "figure.dpi": 110,
    "font.size": 11,
})

# --- Destination: the thesis figures directory. --------------------------------------
THESIS_FIG = Path(r"C:\Users\Karol\Desktop\Magisterka\magisterka-karola-txt\figures")
THESIS_FIG.mkdir(parents=True, exist_ok=True)


def find_project_root(marker: str = "pep-compass", start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    for cand in [here, *here.parents]:
        if cand.name == marker:
            return cand
    for cand in [here, *here.parents]:
        if (cand / "results" / "data" / "all_in").is_dir():
            return cand
    raise FileNotFoundError(f"Could not locate '{marker}' from {here}")


PROJECT_ROOT = find_project_root()
CACHE = PROJECT_ROOT / "results" / "data" / "all_in" / "_cache"


def save(fig, name: str) -> Path:
    """Save ``fig`` as a vector PDF into the thesis figures dir and report the path."""
    out = THESIS_FIG / name
    fig.savefig(out, format="pdf")
    plt.close(fig)
    print(f"  wrote {out}")
    return out


# --- Amino-acid groupings / palettes (verbatim from the notebooks). ------------------
ALL_AA = list("ACDEFGHIKLMNPQRSTVWY")

AMINO_ACID_CLASSES = {
    "polarity": {
        "hydrophobic": ["A", "V", "L", "I", "M", "F", "W", "P", "G"],
        "polar": ["S", "T", "Y", "C", "N", "Q"],
        "charged": ["K", "R", "H", "D", "E"],
    },
    "charge": {
        "positive": ["K", "R", "H"],
        "negative": ["D", "E"],
        "neutral": ["A", "V", "L", "I", "M", "F", "W", "P", "G", "S", "T", "Y", "C", "N", "Q"],
    },
    "chemical_type": {
        "aliphatic": ["G", "A", "V", "L", "I"],
        "aromatic": ["F", "Y", "W"],
        "hydroxyl": ["S", "T", "Y"],
        "acidic": ["D", "E"],
        "amide": ["N", "Q"],
        "basic": ["K", "R", "H"],
        "sulfur": ["C", "M"],
        "imino": ["P"],
    },
}

GREEN_WHITE_RED = LinearSegmentedColormap.from_list(
    "green_white_red",
    ["#1a9641", "#a6d96a", "#ffffff", "#fdae61", "#d7191c"],
)
SOFT_BLUE = LinearSegmentedColormap.from_list(
    "soft_blue",
    ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6"],
)
SOFT_DIVERGING = sns.color_palette("vlag", as_cmap=True)


def _grouped_axis_order(aas, classification):
    if classification is None:
        return list(aas), [], []
    groups = AMINO_ACID_CLASSES[classification]
    sorted_aa, assigned, gsizes, gnames = [], set(), [], []
    for gn, mem in groups.items():
        pres = [aa for aa in aas if aa in mem and aa not in assigned]
        if pres:
            sorted_aa.extend(pres)
            assigned.update(pres)
            gsizes.append(len(pres))
            gnames.append(gn)
    sorted_aa.extend([aa for aa in aas if aa not in assigned])
    return sorted_aa, gsizes, gnames


def _draw_group_bars(ax, df, gsizes, gnames, group_colors=None):
    if not gsizes:
        return
    boundaries = np.cumsum(gsizes)
    for b in boundaries[:-1]:
        ax.axhline(b, color="#9aa0a6", lw=1.0)
        ax.axvline(b, color="#9aa0a6", lw=1.0)
    total = len(df)
    mids = (boundaries - np.array(gsizes) / 2) / total
    sizes_ax = np.array(gsizes) / total
    if group_colors is None:
        palette = sns.color_palette("Set2", len(gnames))
        group_colors = {g: palette[i] for i, g in enumerate(gnames)}
    bar_t = 0.025
    y_gap, x_gap, lab_gap = 0.045, 0.06, 0.012
    for mid, gn, gs in zip(mids, gnames, sizes_ax):
        col = group_colors.get(gn, "lightgray")
        y = 1.0 - mid
        ax.add_patch(plt.Rectangle((-(y_gap + bar_t), y - gs / 2), bar_t, gs,
                                   transform=ax.transAxes, clip_on=False, color=col))
        ax.text(-(y_gap + bar_t) - lab_gap, y, gn, transform=ax.transAxes,
                rotation=90, ha="center", va="center", fontsize=9)
        ax.add_patch(plt.Rectangle((mid - gs / 2, -(x_gap + bar_t)), gs, bar_t,
                                   transform=ax.transAxes, clip_on=False, color=col))
        ax.text(mid, -(x_gap + bar_t) - lab_gap, gn, transform=ax.transAxes,
                rotation=90, ha="center", va="top", fontsize=9)
    ax.yaxis.set_label_coords(-(y_gap + bar_t) - 0.06, 0.5)


def _as_aa_frame(matrix) -> pd.DataFrame:
    if isinstance(matrix, pd.DataFrame):
        return matrix.copy()
    matrix = np.asarray(matrix)
    n = matrix.shape[0]
    labels = ALL_AA if n == len(ALL_AA) else [f"AA{i + 1}" for i in range(n)]
    return pd.DataFrame(matrix, index=labels, columns=labels)


def plot_aa_heatmap(matrix, classification="chemical_type", cmap=GREEN_WHITE_RED,
                    title=None, mask_diagonal=True, vmin=None, vmax=None, center=None,
                    figsize=(9, 8), cbar_label=None, ax=None):
    """Generic grouped AA x AA heatmap (shared by the signature and proposal figures)."""
    df = _as_aa_frame(matrix)
    sorted_aa, gsizes, gnames = _grouped_axis_order(list(df.index), classification)
    df = df.loc[sorted_aa, sorted_aa]

    mask = df.isna()
    if mask_diagonal:
        diag = np.zeros(df.shape, dtype=bool)
        np.fill_diagonal(diag, True)
        mask = mask | diag

    base_cmap = (cmap if isinstance(cmap, Colormap) else plt.get_cmap(cmap)).copy()
    base_cmap.set_bad(color="#ececec")

    created = ax is None
    fig = plt.figure(figsize=figsize) if created else ax.figure
    if created:
        ax = fig.add_subplot(111)
    sns.heatmap(df, cmap=base_cmap, mask=mask, square=True, linewidths=0.5,
                linecolor="white", vmin=vmin, vmax=vmax, center=center,
                cbar_kws={"shrink": 0.7, "aspect": 25, "pad": 0.02,
                          "label": cbar_label or ""}, ax=ax)
    ax.set_title(title or "", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("To amino acid", fontsize=11)
    ax.set_ylabel("From amino acid", fontsize=11)
    ax.tick_params(axis="both", labelsize=9, length=0)
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.setp(ax.get_xticklabels(), rotation=0)
    _draw_group_bars(ax, df, gsizes, gnames)
    return fig, ax
