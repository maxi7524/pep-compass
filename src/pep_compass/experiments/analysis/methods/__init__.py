"""Built-in locality analysis methods."""

from pep_compass.experiments.analysis.methods.distances import (
    add_sequence_distances,
    latent_distance_batches,
)
from pep_compass.experiments.analysis.methods.funnel import retention_summary
from pep_compass.experiments.analysis.methods.filtering import filter_score_summary
from pep_compass.experiments.analysis.methods.outcomes import evaluation_summary
from pep_compass.experiments.analysis.methods.pogs import (
    compute_pogs_distances,
    pogs_acceptance_permutation_test,
    pogs_spearman_by_hamming,
    stratified_candidate_sample,
)
from pep_compass.experiments.analysis.methods.trajectories import reconstruct_trajectory

__all__ = [
    "add_sequence_distances",
    "latent_distance_batches",
    "evaluation_summary",
    "filter_score_summary",
    "compute_pogs_distances",
    "pogs_acceptance_permutation_test",
    "pogs_spearman_by_hamming",
    "reconstruct_trajectory",
    "retention_summary",
    "stratified_candidate_sample",
]
