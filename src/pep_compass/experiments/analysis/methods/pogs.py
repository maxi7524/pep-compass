"""Bounded offline sampling and statistics for PoGS locality validation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pep_compass.experiments.analysis.methods.distances import add_sequence_distances
from pep_compass.experiments.analysis.selection import ExperimentSelection


def stratified_candidate_sample(
    selection: ExperimentSelection,
    candidates_per_stratum: int,
    random_seed: int = 0,
    hamming_values: tuple[int, ...] = (1, 2, 3, 4),
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    """Reservoir-sample equal candidate counts per non-empty PoGS stratum.

    Strata consist of method, parent sequence, grid, LAMS acceptance status,
    and Hamming distance. Memory is bounded by ``candidates_per_stratum`` times
    the number of observed strata.

    :param selection: Deferred LAMS and random-baseline run selection.
    :param candidates_per_stratum: Maximum rows retained per stratum.
    :param random_seed: Seed controlling deterministic reservoir replacement.
    :param hamming_values: Hamming distances eligible for PoGS validation.
    :param chunk_size: Maximum candidate rows read at once.
    :return: Equal-sized samples for every non-empty stratum.
    """
    if candidates_per_stratum < 1:
        raise ValueError("candidates_per_stratum must be positive")
    rng = np.random.default_rng(random_seed)
    reservoirs: dict[tuple, list[dict]] = defaultdict(list)
    seen: dict[tuple, int] = defaultdict(int)
    columns = [
        "sequence",
        "parent_sequence",
        "passed_method_filter",
        "method_score",
        "iteration_id",
        "trajectory_id",
        "step_id",
    ]
    for chunk in selection.scan("candidates", columns=columns, chunk_size=chunk_size):
        chunk = add_sequence_distances(chunk)
        chunk = chunk[chunk["hamming_distance"].isin(hamming_values)]
        for row in chunk.to_dict(orient="records"):
            key = (
                row["method"],
                row["parent_sequence"],
                row["grid_id"],
                bool(row["passed_method_filter"]),
                int(row["hamming_distance"]),
            )
            seen[key] += 1
            reservoir = reservoirs[key]
            if len(reservoir) < candidates_per_stratum:
                reservoir.append(row)
                continue
            replacement = int(rng.integers(seen[key]))
            if replacement < candidates_per_stratum:
                reservoir[replacement] = row
    if not reservoirs:
        return pd.DataFrame()
    equal_count = min(len(rows) for rows in reservoirs.values())
    return pd.DataFrame(
        [row for rows in reservoirs.values() for row in rows[:equal_count]]
    )


def compute_pogs_distances(
    sample: pd.DataFrame,
    encoder_decoder,
    pogs_distance: Callable[[str, str], float],
) -> pd.DataFrame:
    """Compute offline PoGS and proxy distances for a bounded sample.

    :param sample: Output of :func:`stratified_candidate_sample`.
    :param encoder_decoder: Object exposing ``encode_peptides``.
    :param pogs_distance: Project-specific callable that optimizes a PoGS path
        with activity potential disabled and returns its ambient chord sum.
    :return: Sample with PoGS, latent Euclidean, and direct ambient distances.

    The callback keeps this analysis independent from a particular PoGS solver
    configuration while making the required ``lambda=0`` choice explicit at
    the notebook call site.
    """
    result = sample.copy()
    parents = encoder_decoder.encode_peptides(result["parent_sequence"].tolist())
    candidates = encoder_decoder.encode_peptides(result["sequence"].tolist())
    result["latent_euclidean_distance"] = np.linalg.norm(
        candidates.detach().cpu().numpy() - parents.detach().cpu().numpy(), axis=1
    )
    parent_ambient = encoder_decoder.decoder_forward(
        parents, softmax_and_flatten=True
    )
    candidate_ambient = encoder_decoder.decoder_forward(
        candidates, softmax_and_flatten=True
    )
    result["direct_ambient_chord_distance"] = np.linalg.norm(
        candidate_ambient.detach().cpu().numpy()
        - parent_ambient.detach().cpu().numpy(),
        axis=1,
    )
    result["pogs_distance"] = [
        pogs_distance(parent, candidate)
        for parent, candidate in zip(
            result["parent_sequence"], result["sequence"]
        )
    ]
    return result


def pogs_spearman_by_hamming(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute PoGS-to-latent Spearman correlation per Hamming stratum."""
    rows = []
    for distance, group in frame.groupby("hamming_distance", dropna=False):
        statistic = spearmanr(
            group["pogs_distance"], group["latent_euclidean_distance"], nan_policy="omit"
        )
        rows.append(
            {
                "hamming_distance": distance,
                "sample_count": len(group),
                "spearman_correlation": statistic.statistic,
                "p_value": statistic.pvalue,
            }
        )
    return pd.DataFrame(rows)


def pogs_acceptance_permutation_test(
    frame: pd.DataFrame,
    permutations: int = 10_000,
    bootstrap_samples: int = 5_000,
    confidence: float = 0.95,
    random_seed: int = 0,
) -> pd.DataFrame:
    """Compare accepted and rejected PoGS distances within Hamming strata.

    :param frame: Computed PoGS result table.
    :param permutations: Label permutations used for a two-sided test.
    :param bootstrap_samples: Stratified bootstrap samples used for the interval.
    :param confidence: Confidence interval coverage in ``(0, 1)``.
    :param random_seed: Seed for reproducible resampling.
    :return: Median accepted-minus-rejected difference, permutation p-value,
        and percentile confidence interval for every Hamming stratum.
    """
    if permutations < 1 or bootstrap_samples < 1:
        raise ValueError("permutations and bootstrap_samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    rng = np.random.default_rng(random_seed)
    rows = []
    for distance, group in frame.groupby("hamming_distance", dropna=False):
        accepted = group.loc[group["passed_method_filter"], "pogs_distance"].to_numpy()
        rejected = group.loc[~group["passed_method_filter"], "pogs_distance"].to_numpy()
        if not len(accepted) or not len(rejected):
            continue
        observed = float(np.median(accepted) - np.median(rejected))
        combined = np.concatenate([accepted, rejected])
        permutation_differences = np.empty(permutations)
        for index in range(permutations):
            shuffled = rng.permutation(combined)
            permutation_differences[index] = (
                np.median(shuffled[: len(accepted)])
                - np.median(shuffled[len(accepted) :])
            )
        bootstrap_differences = np.empty(bootstrap_samples)
        for index in range(bootstrap_samples):
            bootstrap_differences[index] = (
                np.median(rng.choice(accepted, len(accepted), replace=True))
                - np.median(rng.choice(rejected, len(rejected), replace=True))
            )
        tail = (1.0 - confidence) / 2.0
        rows.append(
            {
                "hamming_distance": distance,
                "accepted_count": len(accepted),
                "rejected_count": len(rejected),
                "median_difference": observed,
                "permutation_p_value": (
                    np.count_nonzero(np.abs(permutation_differences) >= abs(observed))
                    + 1
                )
                / (permutations + 1),
                "confidence_low": np.quantile(bootstrap_differences, tail),
                "confidence_high": np.quantile(
                    bootstrap_differences, 1.0 - tail
                ),
            }
        )
    return pd.DataFrame(rows)
