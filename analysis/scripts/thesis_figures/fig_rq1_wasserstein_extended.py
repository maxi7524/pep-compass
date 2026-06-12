"""RQ1 -- extended all_in Wasserstein clustering of AA->AA substitutions.

This companion to ``fig_rq1_wasserstein.py`` keeps the thesis DBAASP result intact
and reads the broader cached subset built by ``final_wasserstein.ipynb``:
Veltri-negative + Veltri-positive + DBAASP + MIC-data.

It writes:
    rq1_wass_ext_assignment.pdf  -- 20x20 AA->AA cluster-assignment heatmap (k=12)
    rq1_wass_ext_clusters.pdf    -- per-cluster pooled log2 MIC-ratio distributions
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from _common import (PROJECT_ROOT, CACHE, ALL_AA,
                     _grouped_axis_order, _draw_group_bars, save)

sys.path.insert(0, str(PROJECT_ROOT / "analysis"))
from utils.plotting.wasserstein_distance import pairwise_wasserstein  # noqa: E402

DATASET_NAME = "all_in extended subset"
CACHE_TAG = "ext_veltri_negative_veltri_positive_dbaasp_clean_mic_data"
PRIMARY_K = 12
META = ("sequence", "within_dataset_id", "source_file", "dataset")


def derive_substitutions(mutants):
    m = mutants.copy().dropna(subset=["parent", "mutant", "position"])
    m = m[(m["position"] < m["parent"].str.len()) & (m["position"] < m["mutant"].str.len())]
    m = m[m["parent"] != m["mutant"]]
    pos = m["position"].astype(int)
    m = m.assign(from_aa=[s[i] for s, i in zip(m["parent"], pos)],
                 to_aa=[s[i] for s, i in zip(m["mutant"], pos)])
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
    return m, pairs, arrays, np.array(counts)


def relabel_by_benefit(clusters, arrays):
    """Renumber clusters 1..K by ascending pooled mean (most beneficial = 1)."""
    bucket = defaultdict(list)
    for arr, c in zip(arrays, clusters):
        bucket[int(c)].append(arr)
    means = {c: float(np.concatenate(vs).mean()) for c, vs in bucket.items()}
    order = sorted(means, key=lambda c: means[c])
    remap = {old: new for new, old in enumerate(order, start=1)}
    return np.array([remap[int(c)] for c in clusters])


def cluster_colors(clusters):
    uniq = sorted({int(c) for c in clusters})
    palette = sns.color_palette("tab20", n_colors=max(20, len(uniq)))
    return {c: palette[(c - 1) % len(palette)] for c in uniq}, uniq


def plot_cluster_assignment(pairs, clusters, colors, uniq, title):
    mat = np.full((len(ALL_AA), len(ALL_AA)), np.nan)
    idx = {aa: i for i, aa in enumerate(ALL_AA)}
    for (a, b), c in zip(pairs, clusters):
        mat[idx[a], idx[b]] = c
    df = pd.DataFrame(mat, index=ALL_AA, columns=ALL_AA)
    cmap = mcolors.ListedColormap([colors[c] for c in uniq])
    boundaries = np.arange(min(uniq), max(uniq) + 2) - 0.5
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)
    sorted_aa, gsizes, gnames = _grouped_axis_order(list(df.index), "chemical_type")
    df = df.loc[sorted_aa, sorted_aa]
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(df, cmap=cmap, norm=norm, mask=df.isna(), square=True, linewidths=0.5,
                linecolor="white", ax=ax,
                cbar_kws={"shrink": 0.7, "aspect": 25, "pad": 0.02, "label": "Cluster"})
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


def plot_cluster_distributions(arrays, clusters, counts, colors, uniq, title):
    bucket = defaultdict(list)
    for arr, c in zip(arrays, clusters):
        bucket[int(c)].append(arr)
    pooled = {c: np.concatenate(vs) for c, vs in bucket.items()}
    n_pairs = {c: int(np.sum(clusters == c)) for c in uniq}
    n_events = defaultdict(int)
    for cnt, c in zip(counts, clusters):
        n_events[int(c)] += int(cnt)

    all_v = np.concatenate(list(pooled.values()))
    lo, hi = np.percentile(all_v, [1, 99])
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    xr = (lo - pad, hi + pad)

    ncols = 3
    nrows = int(np.ceil(len(uniq) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.6 * nrows),
                             sharex=True)
    axes = np.atleast_2d(axes).ravel()
    for ax, c in zip(axes, uniq):
        vals = pooled[c]
        ax.hist(vals, bins=60, range=xr, color=colors[c], edgecolor="white",
                linewidth=0.4, alpha=0.95, density=True)
        ax.axvline(0.0, color="#9aa0a6", lw=0.8, ls=":")
        ax.axvline(vals.mean(), color="#333", lw=1.0, ls="--")
        ax.set_title(f"Cluster {c}: {n_pairs[c]} pair(s), {n_events[c]:,} events\n"
                     f"mean = {vals.mean():+.3f}", fontsize=9, fontweight="bold")
        ax.set_xlim(*xr)
        ax.tick_params(axis="both", labelsize=8, length=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    for ax in axes[len(uniq):]:
        ax.axis("off")
    fig.supxlabel("log2 (MIC mutant / MIC parent)", fontsize=11)
    fig.supylabel("Density", fontsize=11)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout()
    return fig


def print_summary(pairs, clusters, arrays, counts, uniq):
    print("\ncluster & n_pairs & n_events & mean_log2 & median_log2 & examples")
    for c in uniq:
        members = [f"{a}>{b}" for (a, b), cc in zip(pairs, clusters) if cc == c]
        vals = np.concatenate([arr for arr, cc in zip(arrays, clusters) if cc == c])
        nev = int(sum(cnt for cnt, cc in zip(counts, clusters) if cc == c))
        ex = ", ".join(members[:8]) + (" ..." if len(members) > 8 else "")
        print(f"{c} & {len(members)} & {nev:,} & {vals.mean():+.3f} & "
              f"{np.median(vals):+.3f} & {ex}")


def main():
    parent_cache = CACHE / f"parents_{CACHE_TAG}.parquet"
    mutant_cache = CACHE / f"mutants_{CACHE_TAG}.parquet"
    if not parent_cache.exists() or not mutant_cache.exists():
        raise FileNotFoundError(
            f"Missing extended caches: {parent_cache.name}, {mutant_cache.name}. "
            "Run the extended section of final_wasserstein.ipynb first."
        )

    parents = pd.read_parquet(parent_cache)
    mutants = pd.read_parquet(mutant_cache)
    bact_cols = [c for c in parents.columns if c not in META]
    parent_mic = parents.set_index("sequence")[bact_cols]

    m = derive_substitutions(mutants)
    m, pairs, arrays, sizes = build_pair_arrays(m, parent_mic, bact_cols)
    print(f"[{DATASET_NAME}] events={len(m):,}  pairs={len(pairs)}  "
          f"empty={(sizes == 0).sum()}  low(<5)={(sizes < 5).sum()}")

    t0 = time.time()
    D = pairwise_wasserstein(arrays)
    print(f"[{DATASET_NAME}] pairwise Wasserstein in {time.time()-t0:.1f}s  "
          f"min/mean/max={D.min():.4f}/{D.mean():.4f}/{D.max():.4f}")
    Z = linkage(squareform(D), method="average")
    primary = relabel_by_benefit(fcluster(Z, PRIMARY_K, criterion="maxclust"), arrays)
    colors, uniq = cluster_colors(primary)

    save(plot_cluster_assignment(
        pairs, primary, colors, uniq,
        f"AA$\\to$AA cluster assignment ($1$-Wasserstein, k={PRIMARY_K}, extended all_in subset)"),
        "rq1_wass_ext_assignment.pdf")
    save(plot_cluster_distributions(
        arrays, primary, sizes, colors, uniq,
        f"Per-cluster log2 MIC-ratio distributions (k={PRIMARY_K}, extended all_in subset)"),
        "rq1_wass_ext_clusters.pdf")

    print_summary(pairs, primary, arrays, sizes, uniq)


if __name__ == "__main__":
    main()
