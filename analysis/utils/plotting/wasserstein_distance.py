"""
Fast Wasserstein distance clustering using scipy-like approach (raw values, no binning).
"""

import numpy as np
from typing import List, Dict
from numba import njit, prange
from numba.typed import List as NumbaList


@njit
def _wasserstein_1d_scipy(u_vals: np.ndarray, v_vals: np.ndarray) -> float:
    """
    Fast 1D Wasserstein using scipy's approach: sort values and integrate CDF difference.
    Assumes uniform weights (1/n for each value).

    Args:
        u_vals, v_vals: (N,) sorted values

    Returns:
        Wasserstein distance (scalar)
    """
    # Merge and sort all unique values
    all_vals = np.unique(np.concatenate((u_vals, v_vals)))
    n = len(all_vals)
    if n < 2:
        return 0.0

    # Compute CDFs at each point (uniform weights: count / n)
    cdf_u = np.zeros(n)
    cdf_v = np.zeros(n)

    u_idx = 0
    v_idx = 0

    for i in range(n):
        # Count values <= current point
        while u_idx < len(u_vals) and u_vals[u_idx] <= all_vals[i]:
            u_idx += 1
        while v_idx < len(v_vals) and v_vals[v_idx] <= all_vals[i]:
            v_idx += 1

        cdf_u[i] = u_idx / len(u_vals) if len(u_vals) > 0 else 0.0
        cdf_v[i] = v_idx / len(v_vals) if len(v_vals) > 0 else 0.0

    # Integrate |CDF_u - CDF_v| with actual distances
    distance = 0.0
    for i in range(n - 1):
        width = all_vals[i + 1] - all_vals[i]
        distance += abs(cdf_u[i] - cdf_v[i]) * width

    return distance


@njit
def _preprocess_distribution(vals: np.ndarray):
    """
    Preprocess single distribution: sort values (numba-accelerated).

    Args:
        vals: (V,) array of values

    Returns:
        sorted_vals
    """
    return np.sort(vals.astype(np.float64))


@njit(parallel=True)
def _pairwise_wasserstein_numba(values_list: NumbaList) -> np.ndarray:
    """
    Internal numba function: Compute pairwise Wasserstein distance matrix (parallel).

    Args:
        values_list: NumbaList of M arrays, each (V_i,) containing SORTED values

    Returns:
        (M, M) symmetric distance matrix
    """
    M = len(values_list)
    D = np.zeros((M, M), dtype=np.float64)

    for i in prange(M):
        u_vals = values_list[i]
        for j in range(i + 1, M):
            v_vals = values_list[j]
            d = _wasserstein_1d_scipy(u_vals, v_vals)
            D[i, j] = d
            D[j, i] = d

    return D


def pairwise_wasserstein(values_list: List[np.ndarray]) -> np.ndarray:
    """
    Compute pairwise Wasserstein distance matrix (fully numba-accelerated).

    Preprocessing (sorting) is done in numba, then pairwise computation runs in parallel.
    Assumes uniform weights for all distributions.

    Args:
        values_list: List of M arrays, each (V_i,) containing values for distribution i

    Returns:
        (M, M) symmetric distance matrix
    """
    M = len(values_list)
    # Preprocess all distributions (numba-accelerated)
    processed_vals = []
    for i in range(M):
        sorted_vals = _preprocess_distribution(values_list[i])
        processed_vals.append(sorted_vals)

    # Convert to numba-typed list (fast, just type conversion)
    numba_vals = NumbaList(processed_vals)

    # Call numba-accelerated pairwise computation (parallel)
    return _pairwise_wasserstein_numba(numba_vals)
