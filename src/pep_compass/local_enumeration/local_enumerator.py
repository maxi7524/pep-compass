"""Local candidate enumeration strategies.

The canonical enumerators from ``dev`` remain unchanged. Two composition
classes replace complete trajectory loops copied across historical scripts:

* ``SamplingFilteredMutationLocalEnumerator`` contains the common loop from
  ``upstream/rl_trials:scripts/lebo_plus.py``, ``lpbebo_plus.py``, ``move.py``,
  and ``random_lebo.py``;
* ``FilteredMutationLocalEnumerator`` contains the single-Jacobian LPBEBO path
  from ``upstream/rl_trials:scripts/run_lpbebo_optimization_apex.py``.

Both delegate method-specific candidate selection to ``mutation_filters.py``.
They reuse canonical SORBES from ``sampling_walker.py`` and canonical MUTANG
from ``mutation_enumerator.py`` rather than copying historical implementations.
"""

import logging
from abc import ABC
from collections import defaultdict
from copy import deepcopy
from typing import TYPE_CHECKING

import Levenshtein
import numpy as np
import torch
from joblib import Parallel, delayed, parallel_backend

from pep_compass.local_enumeration.catalog import (
    EnumerationTrace,
    LocalEnumerationCatalog,
)
from pep_compass.local_enumeration.mutation_enumerator import MutationEnumerator
from pep_compass.local_enumeration.sampling_walker import \
    SamplingWalker, SubRiemannianTangentSpace
from pep_compass.models.encoder_decoder.hydramp_encoder_decoder import \
    HydrAMPEncoderDecoder
from pep_compass.utils.sequence_utils import translate_generated_peptide

if TYPE_CHECKING:
    from pep_compass.local_enumeration.mutation.mutation_filters import (
        MutationCandidateFilter,
    )

logger = logging.getLogger(__name__)


def _trace_value(values: np.ndarray | None, index: int) -> float | int | None:
    """Return a scalar trace value when the candidate has score diagnostics."""
    if values is None or index >= len(values):
        return None
    return values[index].item()


def _method_score_map(filter_trace, sequences: list[str]) -> dict[str, tuple]:
    """Return filter diagnostics keyed by their aligned candidate sequence."""
    if filter_trace is None:
        return {}
    return {
        sequence: (
            _trace_value(filter_trace.scores, index),
            _trace_value(filter_trace.probabilities, index),
            _trace_value(filter_trace.cumulative_probabilities, index),
            _trace_value(filter_trace.ranks, index),
        )
        for index, sequence in enumerate(sequences)
    }


class LocalEnumerator(ABC):

    last_trace: EnumerationTrace

    def local_enumeration(self, center_sequence: str) -> set[str]:
        """
        Local enumeration method to be implemented by subclasses.
        This method should return a set containing sequences in the local neighborhood of the center_sequence.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")


class SamplingMutationLocalEnumerator(LocalEnumerator):
    def __init__(
        self,
        encoder_decoder: HydrAMPEncoderDecoder,
        sampling_walker: SamplingWalker,
        mutation_enumerator: MutationEnumerator,
        walker_trajectories_number: int,
        time_walk_budget: float,
        max_neighbour_levenstein: int | None = None,
        device: str = "cpu",
        tracking_level: str = "short",
        store_walker_latents: bool = False,
        stop_walker_at_levenshtein_limit: bool = True,
    ):
        super().__init__()
        self.encoder_decoder = encoder_decoder
        self.sampling_walker = sampling_walker
        self.mutation_enumerator = mutation_enumerator
        self.walker_trajectories_number = walker_trajectories_number
        self.time_walk_budget = time_walk_budget
        self.max_neighbour_levenstein = max_neighbour_levenstein
        self.device = encoder_decoder.device
        self.tracking_level = tracking_level
        self.store_walker_latents = store_walker_latents
        self.stop_walker_at_levenshtein_limit = stop_walker_at_levenshtein_limit

        self.max_neighbour_levenstein = max_neighbour_levenstein or 25
        self.last_trace = EnumerationTrace()

    def local_enumeration(self, center_peptide) -> set[str]:
        neighbor_peptides = set()
        catalog = LocalEnumerationCatalog(
            self.tracking_level, self.store_walker_latents
        )
        self.last_trace = catalog.trace

        with torch.no_grad():
            initial_latent_position = self.encoder_decoder.encode_peptides(
                [center_peptide]
            )[0]

        for trajectory_iter in range(self.walker_trajectories_number):

            current_latent_position = initial_latent_position
            time_walk = 0.0
            current_peptide = center_peptide
            walker_step = 0

            while time_walk < self.time_walk_budget:

                mutation_parent = current_peptide
                previous_latent_position = current_latent_position
                new_latent_position, step_info = self.sampling_walker.step(
                    current_latent_position
                )
                adjusted_time_step = step_info["adjusted_time_step"]
                U = step_info["U"]
                S = step_info["S"]

                mutations = self.mutation_enumerator.get_mutations_from_s_u(S, U)
                mutations = {
                    position: amino_acids
                    for position, amino_acids in mutations.items()
                    if position < len(current_peptide)
                }
                mutated_peptides = list(
                    dict.fromkeys(
                        self.mutation_enumerator.mutate_peptide(
                            current_peptide, mutations
                        )
                    )
                )

                with torch.no_grad():
                    current_peptide = self.encoder_decoder.decode_peptides(
                        new_latent_position
                    )[0]
                current_latent_position = new_latent_position
                time_walk += adjusted_time_step
                walker_step += 1

                new_neighbor_peptides = [
                    peptide
                    for peptide in mutated_peptides
                    if Levenshtein.distance(peptide, center_peptide)
                    <= self.max_neighbour_levenstein
                ]
                tangent_space = step_info["tangent_space"]
                catalog.record_step(
                    trajectory_id=trajectory_iter,
                    step_id=walker_step,
                    parent_sequence=mutation_parent,
                    next_sequence=current_peptide,
                    current_latent_position=previous_latent_position,
                    next_latent_position=new_latent_position,
                    singular_values=S,
                    effective_dimension=tangent_space.horizontal_dim,
                    adjusted_time_step=adjusted_time_step,
                    mutations=mutations,
                    theoretical_product_count=len(mutated_peptides),
                    post_cap_candidates=mutated_peptides,
                    post_method_filter_candidates=mutated_peptides,
                    post_constraint_candidates=new_neighbor_peptides,
                    method_scores={},
                    center_sequence=center_peptide,
                )

                neighbor_peptides.update(new_neighbor_peptides)

                logger.info(
                    f"Trajectory {trajectory_iter} Step {walker_step} Time {time_walk} / {self.time_walk_budget} Levenstain {Levenshtein.distance(current_peptide, center_peptide)}: Found {len(neighbor_peptides)} peptides ."
                )
                if (
                    self.stop_walker_at_levenshtein_limit
                    and Levenshtein.distance(current_peptide, center_peptide)
                    > self.max_neighbour_levenstein
                ):
                    logger.info(
                        f"Reached {Levenshtein.distance(current_peptide, center_peptide)} distance. Stopping trajectory."
                    )
                    break

        return neighbor_peptides


class SamplingFilteredMutationLocalEnumerator(SamplingMutationLocalEnumerator):
    """SORBES/MUTANG local enumeration with a configurable candidate filter.

    This class contains the trajectory loop duplicated by the historical
    ``SamplingWithMutangPlusLocalEnumerator``,
    ``SamplingWithMutangPlusPlusLocalEnumerator``,
    ``SamplingWithMoveLocalEnumerator``, and ``RandomLocalEnumerator`` classes.
    Their method-specific candidate logic now lives in ``mutation_filters.py``.

    Each step advances canonical SORBES, extracts ``U`` and ``S`` from the step,
    asks canonical MUTANG for candidate residue indices, delegates scoring and
    selection to ``candidate_filter``, and finally applies the Levenshtein radius
    relative to the original trajectory center.
    """

    def __init__(
        self,
        *args,
        candidate_filter: "MutationCandidateFilter",
        **kwargs,
    ):
        """Initialize a filtered SORBES/MUTANG enumerator.

        :param args: Positional arguments forwarded to
            :class:`SamplingMutationLocalEnumerator`.
        :param candidate_filter: Policy used to score or select MUTANG
            candidates at every SORBES step.
        :param kwargs: Keyword arguments forwarded to
            :class:`SamplingMutationLocalEnumerator`.
        """
        super().__init__(*args, **kwargs)
        self.candidate_filter = candidate_filter

    def local_enumeration(self, center_peptide: str) -> set[str]:
        """Enumerate filtered candidates along SORBES trajectories.

        :param center_peptide: Peptide used as the trajectory origin and the
            centre of the Levenshtein-radius constraint.
        :return: Unique candidate peptide sequences accepted by the filter and
            radius constraint.
        """
        neighbor_peptides: set[str] = set()
        catalog = LocalEnumerationCatalog(
            self.tracking_level, self.store_walker_latents
        )
        self.last_trace = catalog.trace
        with torch.no_grad():
            initial_latent_position = self.encoder_decoder.encode_peptides(
                [center_peptide]
            )[0]

        for trajectory_iter in range(self.walker_trajectories_number):
            current_latent_position = initial_latent_position
            current_peptide = center_peptide
            time_walk = 0.0
            walker_step = 0
            while time_walk < self.time_walk_budget:
                new_latent_position, step_info = self.sampling_walker.step(
                    current_latent_position
                )
                mutations = self.mutation_enumerator.get_mutations_from_s_u(
                    step_info["S"],
                    step_info["U"],
                )
                mutations = {
                    position: amino_acids
                    for position, amino_acids in mutations.items()
                    if position < len(current_peptide)
                }
                candidates = self.candidate_filter.filter_candidates(
                    current_peptide,
                    mutations,
                    tangent_space=step_info["tangent_space"],
                )
                bounded_candidates = self.candidate_filter.last_bounded_candidates
                accepted = [
                    peptide for peptide in candidates
                    if Levenshtein.distance(peptide, center_peptide)
                    <= self.max_neighbour_levenstein
                ]
                with torch.no_grad():
                    next_peptide = self.encoder_decoder.decode_peptides(
                        new_latent_position
                    )[0]
                step_id = walker_step + 1
                tangent_space = step_info["tangent_space"]
                catalog.record_step(
                    trajectory_id=trajectory_iter,
                    step_id=step_id,
                    parent_sequence=current_peptide,
                    next_sequence=next_peptide,
                    current_latent_position=current_latent_position,
                    next_latent_position=new_latent_position,
                    singular_values=step_info["S"],
                    effective_dimension=tangent_space.horizontal_dim,
                    adjusted_time_step=step_info["adjusted_time_step"],
                    mutations=mutations,
                    theoretical_product_count=getattr(
                        self.candidate_filter,
                        "last_proposed_count",
                        len(bounded_candidates),
                    ),
                    post_cap_candidates=bounded_candidates,
                    post_method_filter_candidates=candidates,
                    post_constraint_candidates=accepted,
                    method_scores=_method_score_map(
                        self.candidate_filter.last_filter_trace,
                        bounded_candidates,
                    ),
                    center_sequence=center_peptide,
                )
                neighbor_peptides.update(accepted)

                current_peptide = next_peptide
                current_latent_position = new_latent_position
                time_walk += step_info["adjusted_time_step"]
                walker_step += 1
                logger.info(
                    "Trajectory %s Step %s Time %s / %s: Found %s peptides.",
                    trajectory_iter,
                    walker_step,
                    time_walk,
                    self.time_walk_budget,
                    len(neighbor_peptides),
                )
                if (
                    self.stop_walker_at_levenshtein_limit
                    and Levenshtein.distance(current_peptide, center_peptide)
                    > self.max_neighbour_levenstein
                ):
                    break
        return neighbor_peptides


class EuclideanWalkerLocalEnumeratorWithAmbientDistance(LocalEnumerator):
    def __init__(
        self,
        encoder_decoder: HydrAMPEncoderDecoder,
        walker_trajectories_number: int,
        max_walker_ambient_distance: float,
        max_neighbour_levenstein: int = None,
        time_step: float = 0.1,
        device: str = "cpu",
    ):
        self.encoder_decoder = encoder_decoder
        self.walker_trajectories_number = walker_trajectories_number
        self.max_walker_ambient_distance = max_walker_ambient_distance
        self.time_step = time_step
        self.max_neighbour_levenstein = max_neighbour_levenstein
        self.device = encoder_decoder.device
        self.max_neighbour_levenstein = max_neighbour_levenstein or 25

    def local_enumeration(self, center_peptide: str) -> set[str]:

        neighbor_peptides = set()

        initial_latent_point = self.encoder_decoder.encode_peptides([center_peptide])[0]
        initial_ambient_point = self.encoder_decoder.decoder_forward(
            initial_latent_point, softmax_and_flatten=True
        )

        logger.info(
            f"Time step: {self.time_step}, Total ambient distance: {self.max_walker_ambient_distance }"
        )

        for i in range(self.walker_trajectories_number):

            walker_ambient_distance = 0.0
            current_latent_point = initial_latent_point.clone()
            previous_ambient_position = initial_ambient_point.clone()

            while walker_ambient_distance <= self.max_walker_ambient_distance:  # = 16.0

                # Euclidean walker robi 20 kroków tak żęby czas wynosił 0.25. CZyli jego time_step = 0.25/20; spatal_step = sqrt(time_step) = sqrt(0.25/20)
                # 0.25 * 64 = 16

                # Generate a random direction in the latent space
                direction = torch.randn(
                    initial_latent_point.shape[-1], device=self.device
                )
                direction = (
                    direction
                    / torch.norm(direction)
                    * np.sqrt(initial_latent_point.shape[-1])  # srqt(64)
                )

                current_latent_point += direction * self.time_step**0.5

                decoder_output = self.encoder_decoder.decoder_forward(
                    current_latent_point, softmax_and_flatten=False
                )
                current_peptide = translate_generated_peptide(decoder_output)

                current_ambient_position = torch.softmax(
                    decoder_output / self.encoder_decoder.temp, dim=-1
                ).flatten()

                ambient_dist = torch.dist(
                    previous_ambient_position, current_ambient_position, p=2
                )
                walker_ambient_distance += ambient_dist.item()

                logger.info(
                    f"Trajectory {i} Ambient distance {walker_ambient_distance:.4f} / {self.max_walker_ambient_distance:.4f} Levenshtein {Levenshtein.distance(current_peptide, center_peptide)}"
                )

                if (
                    Levenshtein.distance(current_peptide, center_peptide)
                    > self.max_neighbour_levenstein
                ):
                    logger.info(
                        f"Reached {Levenshtein.distance(current_peptide, center_peptide)} distance. Stopping trajectory."
                    )
                    break
                neighbor_peptides.add(current_peptide)

        return neighbor_peptides


class EuclideanWalkerLocalEnumerator(LocalEnumerator):
    def __init__(
        self,
        encoder_decoder: HydrAMPEncoderDecoder,
        walker_trajectories_number: int,
        walker_time: float,
        max_neighbour_levenstein: int = None,
        time_step: float = 0.1,
        batch_size: int = 1000,
        device: str = "cpu",
    ):
        self.encoder_decoder = encoder_decoder
        self.walker_trajectories_number = walker_trajectories_number
        self.walker_time = walker_time
        self.device = encoder_decoder.device
        self.time_step = time_step
        self.max_neighbour_levenstein = max_neighbour_levenstein or 25
        self.batch_size = batch_size

    def local_enumeration(self, center_peptide) -> set[str]:
        neighbor_peptides = set()

        initial_latent_point = self.encoder_decoder.encode_peptides([center_peptide])[0]

        number_of_steps = int(self.walker_time / self.time_step)

        logger.info(
            f"Number of steps: {number_of_steps}, Time step: {self.time_step}, Total time: {self.walker_time}"
        )

        normal_sample = torch.randn(
            (
                self.walker_trajectories_number,
                number_of_steps,
                initial_latent_point.shape[-1],
            ),
            device=self.device,
        )

        normal_sample = (
            normal_sample
            / torch.norm(normal_sample, dim=-1, keepdim=True)
            # * torch.sqrt(Sigma z enkodera)
            * self.time_step**0.5
            * np.sqrt(initial_latent_point.shape[-1])
        )

        latent_trajectories = initial_latent_point + torch.cumsum(normal_sample, dim=1)

        batched_latent_trajectories = latent_trajectories.reshape(
            self.walker_trajectories_number * number_of_steps,
            initial_latent_point.shape[-1],
        )

        with torch.no_grad():
            decoded_peptides = self.encoder_decoder.decode_peptides(
                batched_latent_trajectories, batch_size=self.batch_size
            )

        neighbor_peptides.add(decoded_peptides)

        return neighbor_peptides


class NormalSamplingLocalEnumerator(LocalEnumerator):

    def __init__(
        self,
        encoder_decoder: HydrAMPEncoderDecoder,
        max_neighbour_levenstein: int = None,
        number_of_samples: int = 1000,
        sampling_temperature: float = 1.0,
        batch_size: int = 5000,
        device: str = "cpu",
    ):
        self.max_neighbour_levenstein = max_neighbour_levenstein or 25
        self.batch_size = batch_size
        self.encoder_decoder = encoder_decoder
        self.device = encoder_decoder.device

        self.sampling_temperature = sampling_temperature
        self.number_of_samples = number_of_samples

    def local_enumeration(self, center_peptide) -> set[str]:

        mean, std = self.encoder_decoder.encode_peptides_with_std([center_peptide])
        std = torch.exp(std / 2)
        mean, std = mean[0], std[0]  # (64, ), (64, )

        sampled_latent_points = (
            torch.randn((self.number_of_samples, mean.shape[0]), device=self.device)
            * std
            * self.sampling_temperature
            + mean
        )  # (64, )

        generated_peptides = self.encoder_decoder.decode_peptides(
            sampled_latent_points, batch_size=self.batch_size
        )

        neighbour_peptides = {
            peptide
            for peptide in generated_peptides
            if Levenshtein.distance(peptide, center_peptide)
            <= self.max_neighbour_levenstein
        }

        return neighbour_peptides


class MutationLocalEnumerator(LocalEnumerator):
    def __init__(
        self,
        encoder_decoder: HydrAMPEncoderDecoder,
        mutation_generator: MutationEnumerator,
        max_neighbour_levenstein: int = None,
        device: str = "cpu",
        tracking_level: str = "short",
        store_walker_latents: bool = False,
    ):
        self.encoder_decoder = encoder_decoder
        self.mutation_generator = mutation_generator
        self.device = encoder_decoder.device
        self.tracking_level = tracking_level
        self.store_walker_latents = store_walker_latents
        self.max_neighbour_levenstein = max_neighbour_levenstein

        if self.max_neighbour_levenstein is None:
            self.max_neighbour_levenstein = 25
        self.last_trace = EnumerationTrace()

    def local_enumeration(self, center_peptide, **kwargs) -> set[str]:

        with torch.no_grad():
            # Encode the center peptide to get the latent point
            center_latent_point = self.encoder_decoder.encode_peptides([center_peptide])

        with torch.no_grad():
            jacobian = self.encoder_decoder.decoder_jacobian(center_latent_point)[0]
            logger.debug(f"jacobian device: {jacobian.device}")
            U, S, V = torch.linalg.svd(jacobian, full_matrices=False)

            logger.debug(f"U, S, V device: {U.device}, {S.device}, {V.device}")

        mutated_peptides = self.mutation_generator.mutate(center_peptide, U=U, S=S)

        neighbor_peptides = {
            peptide
            for peptide in mutated_peptides
            if Levenshtein.distance(peptide, center_peptide)
            <= self.max_neighbour_levenstein
        }

        return neighbor_peptides


class FilteredMutationLocalEnumerator(MutationLocalEnumerator):
    """Single-point MUTANG enumeration with a configurable candidate filter.

    Unlike ``SamplingFilteredMutationLocalEnumerator``, this path does not walk
    with SORBES. It computes the decoder Jacobian at the optimization center,
    obtains MUTANG choices from its SVD, applies the configured filter once, and
    enforces the Levenshtein radius. The runner uses it for LPBEBO.
    """

    def __init__(
        self,
        *args,
        candidate_filter: "MutationCandidateFilter",
        **kwargs,
    ):
        """Initialize a filtered single-Jacobian MUTANG enumerator.

        :param args: Positional arguments forwarded to
            :class:`MutationLocalEnumerator`.
        :param candidate_filter: Policy used to select candidates from the
            MUTANG mutation map.
        :param kwargs: Keyword arguments forwarded to
            :class:`MutationLocalEnumerator`.
        """
        super().__init__(*args, **kwargs)
        self.candidate_filter = candidate_filter

    def local_enumeration(self, center_peptide: str, **kwargs) -> set[str]:
        """Enumerate candidates from one decoder Jacobian at the centre.

        :param center_peptide: Peptide at which the Jacobian and SVD are
            evaluated.
        :param kwargs: Reserved for compatibility with the local-enumerator
            protocol.
        :return: Unique filtered candidates inside the configured radius.
        """
        with torch.no_grad():
            center_latent_point = self.encoder_decoder.encode_peptides([center_peptide])
            jacobian = self.encoder_decoder.decoder_jacobian(center_latent_point)[0]
            left, singular_values, right = torch.linalg.svd(
                jacobian, full_matrices=False
            )
        tangent_space = SubRiemannianTangentSpace(
            U=left,
            S=singular_values,
            V=right,
            horizontal_threshold=getattr(
                self.candidate_filter, "horizontal_threshold", 0.1
            ),
            device=str(center_latent_point.device),
        )
        mutations = self.mutation_generator.get_mutations_from_s_u(
            singular_values, left
        )
        mutations = {
            position: amino_acids
            for position, amino_acids in mutations.items()
            if position < len(center_peptide)
        }
        candidates = self.candidate_filter.filter_candidates(
            center_peptide, mutations, tangent_space=tangent_space
        )
        accepted = {
            peptide
            for peptide in candidates
            if Levenshtein.distance(peptide, center_peptide)
            <= self.max_neighbour_levenstein
        }
        bounded_candidates = self.candidate_filter.last_bounded_candidates
        catalog = LocalEnumerationCatalog(
            self.tracking_level, self.store_walker_latents
        )
        catalog.record_step(
            trajectory_id=0,
            step_id=1,
            parent_sequence=center_peptide,
            next_sequence=center_peptide,
            current_latent_position=center_latent_point,
            next_latent_position=center_latent_point,
            singular_values=singular_values,
            effective_dimension=tangent_space.horizontal_dim,
            adjusted_time_step=0.0,
            mutations=mutations,
            theoretical_product_count=getattr(
                self.candidate_filter, "last_proposed_count", len(bounded_candidates)
            ),
            post_cap_candidates=bounded_candidates,
            post_method_filter_candidates=candidates,
            post_constraint_candidates=list(accepted),
            method_scores=_method_score_map(
                self.candidate_filter.last_filter_trace,
                bounded_candidates,
            ),
            center_sequence=center_peptide,
        )
        self.last_trace = catalog.trace
        return accepted


# Max: Kept the legacy multi-walker disabled | loky copies model state and tensors between processes.
# class MultiWalkerLocalEnumerator(LocalEnumerator):
#     def __init__(
#         self,
#         encoder_decoder: HydrAMPEncoderDecoder,
#         walker_factory: RiemannianWalkerFactory,
#         walker_trajectories_number: int,
#         walker_time: float,
#         max_neighbour_levenstein: int = None,
#         max_neighbour_levenstein_frac: float = None,
#         device: str = "cpu",
#     ):
#         super().__init__(max_neighbour_levenstein, max_neighbour_levenstein_frac)
#         self.encoder_decoder = encoder_decoder
#         self.walker_factory = walker_factory
#         self.walker_trajectories_number = walker_trajectories_number
#         self.walker_time = walker_time
#         self.device = device

#     def reset(self):
#         pass

#     def walk_trajectory(
#         self,
#         center_peptide,
#         initial_latent_point,
#         initial_latent_velocity,
#         max_neighbour_levenstein,
#     ) -> set[str]:
#         peptides_in_trajectory = set()

#         walker = self.walker_factory.create()
#         step_peptide = center_peptide

#         walker_step = 0
#         while walker.time < self.walker_time:
#             walker.step()

#             current_latent_point = walker.current_step.latent_position

#             step_peptide = translate_generated_peptide(
#                 self.encoder_decoder.decoder_forward(
#                     current_latent_point.to(self.device),
#                     softmax_and_flatten=False,
#                 )
#             )
#             logger.info(
#                 f"Step {walker_step} Time {walker.time} / {self.walker_time} Levenstain {Levenshtein.distance(step_peptide, center_peptide)}"
#             )
#             if (
#                 Levenshtein.distance(step_peptide, center_peptide)
#                 > max_neighbour_levenstein
#             ):
#                 logger.info(
#                     f"Reached {Levenshtein.distance(step_peptide, center_peptide)} distance. Stopping trajectory."
#                 )
#                 break

#             walker_step += 1

#         return peptides_in_trajectory

#     def local_enumeration(self, center_peptide, **kwargs) -> set[]:

#         # TODO: refactor this part
#         if self.max_neighbour_levenstein is None:
#             max_neighbour_levenstein = len(center_peptide)
#         else:
#             max_neighbour_levenstein = self.max_neighbour_levenstein
#         if self.max_neighbour_levenstein_frac is not None:
#             max_neighbour_levenstein = int(
#                 self.max_neighbour_levenstein_frac * len(center_peptide)
#             )

#         neighbor_peptides = set()

#         initial_latent_point = self.encoder_decoder.encode_peptides([center_peptide])[0]
#         initial_latent_velocity = torch.zeros_like(initial_latent_point)

#         neighborhood = Neigborhood(
#             center_peptide=center_peptide,
#             trajectories=[],
#             # score=self.scored_peptides[center_peptide],
#         )

#         with parallel_backend("loky"):
#             trajectories = Parallel(n_jobs=4)(
#                 delayed(self.walk_trajectory)(
#                     center_peptide,
#                     initial_latent_point,
#                     initial_latent_velocity,
#                     max_neighbour_levenstein,
#                 )
#                 for _ in range(self.walker_trajectories_number)
#             )

#         for trajectory in trajectories:
#             # Add the trajectory to the neighborhood
#             neighborhood.trajectories.append(trajectory)

#             # Collect unique peptides from the trajectory
#             for step in trajectory.steps:
#                 neighbor_peptides.add(step.peptide)

#             logger.info(f"Trajectory completed with {len(trajectory.steps)} steps.")

#         return neighborhood
