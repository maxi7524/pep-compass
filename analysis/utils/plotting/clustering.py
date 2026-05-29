from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict


def hierarchical_cluster(D: np.ndarray, method: str = "average") -> np.ndarray:
    """
    Perform hierarchical clustering.

    Args:
        D: (M, M) symmetric distance matrix

    Returns:
        (M-1, 4) linkage matrix Z
    """
    return linkage(squareform(D), method=method)


def get_clusters_at_levels(Z: np.ndarray, levels: List[int]) -> Dict[int, np.ndarray]:
    """
    Get cluster assignments at multiple hierarchy levels.

    Args:
        Z: (M-1, 4) linkage matrix
        levels: list of cluster counts, e.g. [5, 10, 20]

    Returns:
        {level: (M,) cluster labels} for each level
    """
    return {k: fcluster(Z, k, criterion="maxclust") for k in levels}


def compute_centroids(
    values_list: List[np.ndarray], clusters: np.ndarray
) -> Dict[int, np.ndarray]:
    """
    Compute mean distribution (centroid) for each cluster.

    For each cluster, concatenates all values. This approximates the Wasserstein barycenter
    and works well for clustering.

    Args:
        values_list: List of M arrays, each (V_i,) containing values
        clusters: (M,) cluster labels

    Returns:
        {cluster_id: (V,) values array} for each cluster
    """
    centroids = {}
    for cluster_id in np.unique(clusters):
        cluster_values = [
            values_list[i] for i in range(len(values_list)) if clusters[i] == cluster_id
        ]

        if len(cluster_values) > 0:
            all_vals = np.concatenate(cluster_values)
            centroids[cluster_id] = all_vals

    return centroids


def centroids_at_levels(
    values_list: List[np.ndarray], Z: np.ndarray, levels: List[int]
) -> Dict[int, Dict[int, np.ndarray]]:
    """
    Compute centroids at multiple clustering levels.

    Args:
        values_list: List of M arrays, each (V_i,) containing values
        Z: (M-1, 4) linkage matrix
        levels: list of cluster counts

    Returns:
        {level: {cluster_id: (V,) values array}} for each level
    """
    clusters_by_level = get_clusters_at_levels(Z, levels)
    return {k: compute_centroids(values_list, clusters_by_level[k]) for k in levels}


def plot_clustermap(
    D: np.ndarray,
    Z: np.ndarray,
    clusters: np.ndarray,
    figsize: tuple = (10, 10),
):
    """
    Plot distance matrix as clustermap with cluster color annotations.

    Args:
        D: (M, M) distance matrix
        Z: (M-1, 4) linkage matrix
        clusters: (M,) cluster labels for coloring
    """
    n_clusters = len(np.unique(clusters))
    palette = sns.color_palette("tab20", n_colors=max(20, n_clusters))
    row_colors = [palette[(c - 1) % len(palette)] for c in clusters]

    g = sns.clustermap(
        D,
        row_linkage=Z,
        col_linkage=Z,
        cmap="viridis",
        row_colors=row_colors,
        col_colors=row_colors,
        figsize=figsize,
    )
    plt.setp(g.ax_heatmap.get_xticklabels(), visible=False)
    plt.setp(g.ax_heatmap.get_yticklabels(), visible=False)
    plt.show()
    return g
