"""RQ1 -- Wasserstein-distance clustering of AA->AA substitutions on MIC deltas.

Replicates the 1-Wasserstein clustering of final_wasserstein.ipynb (pooled per-event
log2 MIC-delta distributions per substitution) from the parquet caches and writes vector
PDFs:
    rq1_wass_clustermap.pdf  -- distance clustermap with dendrograms
    rq1_wass_assignment.pdf  -- 20x20 AA->AA cluster-assignment heatmap
    rq1_wass_mds.pdf         -- MDS embedding of the distances

Prints the cluster summary numbers used in the thesis tables.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.manifold import MDS

from _common import (PROJECT_ROOT, CACHE, ALL_AA, SOFT_BLUE,
                     _grouped_axis_order, _draw_group_bars, save)

sys.path.insert(0, str(PROJECT_ROOT / "analysis"))
from utils.plotting.wasserstein_distance import pairwise_wasserstein  # noqa: E402

LEVELS = [6, 12, 20]
PRIMARY_K = 12
MIN_EVENTS_PAIR = 5


def derive_substitutions(mutants, parent_mic, bact_cols):
    m = mutants.copy().dropna(subset=["parent", "mutant", "position"])
    m = m[(m["position"] < m["parent"].str.len()) & (m["position"] < m["mutant"].str.len())]
    m = m[m["parent"] != m["mutant"]]
    pos = m["position"].astype(int)
    m = m.assign(from_aa=[s[i] for s, i in zip(m["parent"], pos)],
                 to_aa=[s[i] for s, i in zip(m["mutant"], pos)],
                 plen=m["parent"].str.len())
    return m[m["from_aa"] != m["to_aa"]].reset_index(drop=True)


def build_pair_arrays(m, parent_mic, bact_cols):
    mut_mic = m[bact_cols].to_numpy(dtype=np.float64)
    par_mic = parent_mic.loc[m["parent"].to_numpy()].to_numpy(dtype=np.float64)
    valid = (mut_mic > 0).all(axis=1) & (par_mic > 0).all(axis=1)
    m = m.loc[valid].reset_index(drop=True)
    deltas = np.log2(mut_mic[valid]) - np.log2(par_mic[valid])
    groups = m.groupby(["from_aa", "to_aa"], sort=False).indices
    pairs, arrays, counts = [], [], []
    for a in ALL_AA:
        for b in ALL_AA:
            if a == b:
                continue
            idx = groups.get((a, b))
            if idx is None or len(idx) == 0:
                arr, n = np.array([0.0]), 0
            else:
                arr, n = deltas[idx].reshape(-1), len(idx)
            pairs.append((a, b))
            arrays.append(arr.astype(np.float64))
            counts.append(n)
    return m, deltas, pairs, arrays, np.array(counts)


def plot_cluster_assignment(pairs, clusters, title):
    A = len(ALL_AA)
    mat = np.full((A, A), np.nan)
    idx = {aa: i for i, aa in enumerate(ALL_AA)}
    for (a, b), c in zip(pairs, clusters):
        mat[idx[a], idx[b]] = c
    df = pd.DataFrame(mat, index=ALL_AA, columns=ALL_AA)
    uniq = sorted({int(c) for c in clusters})
    palette = sns.color_palette("tab20", n_colors=max(20, len(uniq)))
    cmap = mcolors.ListedColormap(palette[:len(uniq)])
    boundaries = np.arange(min(uniq), max(uniq) + 2) - 0.5
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)
    sorted_aa, gsizes, gnames = _grouped_axis_order(list(df.index), "chemical_type")
    df = df.loc[sorted_aa, sorted_aa]
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(df, cmap=cmap, norm=norm, mask=df.isna(), square=True, linewidths=0.5,
                linecolor="white", ax=ax,
                cbar_kws={"shrink": 0.7, "aspect": 25, "pad": 0.02, "label": "Cluster ID"})
    cbar = ax.collections[0].colorbar
    cbar.set_ticks(uniq)
    cbar.set_ticklabels([str(c) for c in uniq])
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("To amino acid", fontsize=11)
    ax.set_ylabel("From amino acid", fontsize=11)
    ax.tick_params(axis="both", labelsize=9, length=0)
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.setp(ax.get_xticklabels(), rotation=0)
    _draw_group_bars(ax, df, gsizes, gnames)
    return fig


def plot_clustermap(D, pairs, Z, clusters_by_level, title, cbar_label):
    dist_df = pd.DataFrame(D, index=[f"{a}>{b}" for a, b in pairs],
                           columns=[f"{a}>{b}" for a, b in pairs])
    row_colors = []
    for k in LEVELS:
        cl = clusters_by_level[k]
        palette = sns.color_palette("tab20", n_colors=max(20, len(set(cl))))
        cmap_k = {c: palette[i % len(palette)] for i, c in enumerate(sorted(set(cl)))}
        row_colors.append(pd.Series([cmap_k[c] for c in cl], name=f"k={k}"))
    g = sns.clustermap(dist_df, row_linkage=Z, col_linkage=Z, cmap=SOFT_BLUE,
                       row_colors=row_colors, col_colors=row_colors, figsize=(11, 11),
                       xticklabels=False, yticklabels=False,
                       cbar_kws={"label": cbar_label}, linewidths=0)
    g.ax_heatmap.set_xlabel("AA->AA pair", fontsize=11)
    g.ax_heatmap.set_ylabel("AA->AA pair", fontsize=11)
    g.figure.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    return g.figure


def plot_mds(D, pairs, clusters, title):
    coords = MDS(n_components=2, dissimilarity="precomputed", random_state=0,
                 n_init=4, normalized_stress="auto").fit_transform(D)
    palette = sns.color_palette("tab20", n_colors=max(20, len(set(clusters))))
    color = {c: palette[(c - 1) % len(palette)] for c in sorted(set(clusters))}
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coords[:, 0], coords[:, 1], c=[color[c] for c in clusters], s=70,
               edgecolor="white", linewidth=0.6, alpha=0.95)
    for c in sorted(set(clusters)):
        mask = clusters == c
        sub = coords[mask]
        i = np.where(mask)[0][np.argmin(((sub - sub.mean(0)) ** 2).sum(1))]
        a, b = pairs[i]
        ax.annotate(f"{a}->{b} (c{c})", (coords[i, 0], coords[i, 1]),
                    xytext=(5, 4), textcoords="offset points", fontsize=8, color="#333")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("MDS-1", fontsize=11)
    ax.set_ylabel("MDS-2", fontsize=11)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    handles = [mpatches.Patch(color=color[c], label=f"c{c}") for c in sorted(set(clusters))]
    ax.legend(handles=handles, title="Cluster", loc="center left",
              bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9, title_fontsize=10)
    fig.tight_layout()
    return fig


def main():
    parents = pd.read_parquet(CACHE / "parents_hydramp_veltri_negative.parquet")
    META = ("sequence", "within_dataset_id", "source_file", "dataset")
    bact_cols = [c for c in parents.columns if c not in META]
    parent_mic = parents.set_index("sequence")[bact_cols]
    mutants = pd.read_parquet(CACHE / "mutants_hydramp_veltri_negative.parquet")
    m = derive_substitutions(mutants, parent_mic, bact_cols)
    m, deltas, pairs, arrays, sizes = build_pair_arrays(m, parent_mic, bact_cols)
    print(f"[1D] events={len(m):,}  pairs={len(pairs)}  empty={(sizes==0).sum()}")

    t0 = time.time()
    D = pairwise_wasserstein(arrays)
    print(f"[1D] pairwise Wasserstein in {time.time()-t0:.1f}s  "
          f"min/mean/max={D.min():.4f}/{D.mean():.4f}/{D.max():.4f}")
    Z = linkage(squareform(D), method="average")
    clusters_by_level = {k: fcluster(Z, k, criterion="maxclust") for k in LEVELS}
    primary = clusters_by_level[PRIMARY_K]

    save(plot_clustermap(D, pairs, Z, clusters_by_level,
                         "Wasserstein clustermap of AA->AA substitutions (Veltri-negative)",
                         "1-Wasserstein on log2 MIC ratio"), "rq1_wass_clustermap.pdf")
    save(plot_cluster_assignment(pairs, primary,
                                 f"AA->AA cluster assignment (1-Wasserstein, k={PRIMARY_K})"),
         "rq1_wass_assignment.pdf")
    save(plot_mds(D, pairs, primary, f"MDS of Wasserstein distances (k={PRIMARY_K})"),
         "rq1_wass_mds.pdf")


if __name__ == "__main__":
    main()
