import logging
from collections import defaultdict
from copy import deepcopy

import Levenshtein
import numpy as np
import torch
from joblib import Parallel, delayed, parallel_backend
from memory_profiler import profile as mem_profile

from database.schema import Neigborhood, Step, Trajectory
from generation.mutation_generator import MutationGenerator
from generation.proteins import to_one_hot, translate_generated_peptide
from generation.riemannian_walker import (AccelerationSource,
                                          RepulsiveAccelerationSource,
                                          RiemannianWalker,
                                          RiemannianWalkerFactory, Step as RiemannianWalkerStep)
from pep_compass.models.encoder_decoder.hydramp_encoder_decoder import HydrAMPEncoderDecoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalEnumerator:
    def __init__(
        self,
        max_neighbour_levenstein: int,
        max_neighbour_levenstein_frac: float,
    ):

        self.max_neighbour_levenstein = max_neighbour_levenstein
        self.max_neighbour_levenstein_frac = max_neighbour_levenstein_frac

    def local_enumeration(self, center_sequence, **kwargs) -> list[str]:
        """
        Local enumeration method to be implemented by subclasses.
        This method should return a list containing sequences in the local neighborhood of the center_sequence.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")


class EuclideanWalkerLocalEnumeratorWithAmbientDistance(LocalEnumerator):
    def __init__(
        self,
        encoder_decoder: HydrAMPEncoderDecoder,
        walker_trajectories_number: int,
        max_walker_ambient_distance: float,
        max_neighbour_levenstein: int = None,
        max_neighbour_levenstein_frac: float = None,
        time_step: float = 0.1,
        device: str = "cpu",
    ):
        super().__init__(max_neighbour_levenstein, max_neighbour_levenstein_frac)
        self.encoder_decoder = encoder_decoder
        self.walker_trajectories_number = walker_trajectories_number
        self.max_walker_ambient_distance = max_walker_ambient_distance
        self.device = device
        self.time_step = time_step

    def local_enumeration(self, center_peptide, batch_size=1000, **kwargs) -> Neigborhood:
        # TODO: refactor this part
        if self.max_neighbour_levenstein is None:
            max_neighbour_levenstein = len(center_peptide)
        else:
            max_neighbour_levenstein = self.max_neighbour_levenstein
        if self.max_neighbour_levenstein_frac is not None:
            max_neighbour_levenstein = int(
                self.max_neighbour_levenstein_frac * len(center_peptide)
            )

        neighbor_peptides = set()

        initial_latent_point = self.encoder_decoder.encode_peptides([center_peptide])[0]
        initial_ambient_point = self.encoder_decoder.decoder_forward(
            initial_latent_point, softmax_and_flatten=True
        )

        neighborhood = Neigborhood(
            center_peptide=center_peptide,
            trajectories=[],
            # score=self.scored_peptides[center_peptide],
        )

        logger.info(
            f"Time step: {self.time_step}, Total ambient distance: {self.max_walker_ambient_distance }"
        )

        for i in range(self.walker_trajectories_number):
            trajectory = Trajectory(
                steps=[],
                starting_peptide=center_peptide,
                starting_point=initial_latent_point.tolist(),
            )

            walker_ambient_distance = 0.0
            current_latent_point = initial_latent_point.clone()
            previous_ambient_position = initial_ambient_point.clone()

            while walker_ambient_distance <= self.max_walker_ambient_distance: # = 16.0
                
                # Euclidean walker robi 20 kroków tak żęby czas wynosił 0.25. CZyli jego time_step = 0.25/20; spatal_step = sqrt(time_step) = sqrt(0.25/20)
                # 0.25 * 64 = 16 

                # Generate a random direction in the latent space
                direction = torch.randn(
                    initial_latent_point.shape[-1], device=self.device
                )
                direction = (
                    direction
                    / torch.norm(direction)
                    * np.sqrt(initial_latent_point.shape[-1]) # srqt(64)
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

                step = Step(
                    peptide=current_peptide,
                    latent_position=current_latent_point.tolist(),
                    adjusted_time_step=self.time_step,
                )
                logger.info(
                    f"Trajectory {i} Ambient distance {walker_ambient_distance:.4f} / {self.max_walker_ambient_distance:.4f} Levenshtein {Levenshtein.distance(current_peptide, center_peptide)}"
                )

                if (
                    Levenshtein.distance(current_peptide, center_peptide)
                    > max_neighbour_levenstein
                ):
                    logger.info(
                        f"Reached {Levenshtein.distance(current_peptide, center_peptide)} distance. Stopping trajectory."
                    )
                    break

                trajectory.steps.append(step)
                neighbor_peptides.add(current_peptide)

            neighborhood.trajectories.append(trajectory)
        neighborhood.neighbor_peptides = list(neighbor_peptides)

        return neighborhood


class EuclideanWalkerLocalSearcher(LocalEnumerator):
    def __init__(
        self,
        encoder_decoder: HydrAMPEncoderDecoder,
        walker_trajectories_number: int,
        walker_time: float,
        max_neighbour_levenstein: int = None,
        max_neighbour_levenstein_frac: float = None,
        time_step: float = 0.1,
        device: str = "cpu",
    ):
        super().__init__(max_neighbour_levenstein, max_neighbour_levenstein_frac)
        self.encoder_decoder = encoder_decoder
        self.walker_trajectories_number = walker_trajectories_number
        self.walker_time = walker_time
        self.device = device
        self.time_step = time_step

    def reset(self):
        pass

    def local_enumeration(self, center_peptide, batch_size=1000, **kwargs) -> Neigborhood:
        # TODO: refactor this part
        if self.max_neighbour_levenstein is None:
            max_neighbour_levenstein = len(center_peptide)
        else:
            max_neighbour_levenstein = self.max_neighbour_levenstein
        if self.max_neighbour_levenstein_frac is not None:
            max_neighbour_levenstein = int(
                self.max_neighbour_levenstein_frac * len(center_peptide)
            )

        neighbor_peptides = set()

        initial_latent_point = self.encoder_decoder.encode_peptides([center_peptide])[0]

        neighborhood = Neigborhood(
            center_peptide=center_peptide,
            trajectories=[],
            # score=self.scored_peptides[center_peptide],
        )

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
                batched_latent_trajectories, batch_size=batch_size
            )

        peptide_trajectories = np.array(decoded_peptides).reshape(
            self.walker_trajectories_number, number_of_steps
        )

        for i in range(self.walker_trajectories_number):
            trajectory = Trajectory(
                steps=[],
                starting_peptide=center_peptide,
                starting_point=initial_latent_point.tolist(),
            )

            for j in range(number_of_steps):
                if (
                    Levenshtein.distance(peptide_trajectories[i][j], center_peptide)
                    > max_neighbour_levenstein
                ):
                    logger.info(
                        f"Reached {Levenshtein.distance(peptide_trajectories[i][j], center_peptide)} distance. Stopping trajectory at step {j}."
                    )
                    break
                step = Step(
                    peptide=peptide_trajectories[i][j],
                    latent_position=latent_trajectories[i, j].tolist(),
                    adjusted_time_step=self.time_step,
                )
                trajectory.steps.append(step)
                neighbor_peptides.add(peptide_trajectories[i][j])

            neighborhood.trajectories.append(trajectory)
        neighborhood.neighbor_peptides = list(neighbor_peptides)

        return neighborhood


class MultiWalkerLocalSearcher(LocalEnumerator):
    def __init__(
        self,
        encoder_decoder: HydrAMPEncoderDecoder,
        walker_factory: RiemannianWalkerFactory,
        walker_trajectories_number: int,
        walker_time: float,
        max_neighbour_levenstein: int = None,
        max_neighbour_levenstein_frac: float = None,
        device: str = "cpu",
    ):
        super().__init__(max_neighbour_levenstein, max_neighbour_levenstein_frac)
        self.encoder_decoder = encoder_decoder
        self.walker_factory = walker_factory
        self.walker_trajectories_number = walker_trajectories_number
        self.walker_time = walker_time
        self.device = device

    def reset(self):
        pass

    def walk_trajectory(
        self,
        center_peptide,
        initial_latent_point,
        initial_latent_velocity,
        max_neighbour_levenstein,
    ) -> Trajectory:
        trajectory = Trajectory(
            steps=[],
            starting_peptide=center_peptide,
            starting_point=initial_latent_point.tolist(),
        )

        walker = self.walker_factory.create()

        walker.reset(
            initial_latent_position=initial_latent_point,
            initial_latent_velocity=initial_latent_velocity,
        )

        step_peptide = center_peptide

        walker_step = 0
        while walker.time < self.walker_time:
            walker.step()

            step = Step.from_riemannian_walker_step(walker.current_step)

            current_latent_point = walker.current_step.latent_position

            step_peptide = translate_generated_peptide(
                self.encoder_decoder.decoder_forward(
                    current_latent_point.to(self.device),
                    softmax_and_flatten=False,
                )
            )
            step.peptide = step_peptide

            logger.info(
                f"Step {walker_step} Time {walker.time} / {self.walker_time} Levenstain {Levenshtein.distance(step_peptide, center_peptide)}"
            )
            if (
                Levenshtein.distance(step_peptide, center_peptide)
                > max_neighbour_levenstein
            ):
                logger.info(
                    f"Reached {Levenshtein.distance(step_peptide, center_peptide)} distance. Stopping trajectory."
                )
                break

            walker_step += 1
            trajectory.steps.append(step)

        return trajectory

    def local_enumeration(self, center_peptide, **kwargs) -> Neigborhood:

        # TODO: refactor this part
        if self.max_neighbour_levenstein is None:
            max_neighbour_levenstein = len(center_peptide)
        else:
            max_neighbour_levenstein = self.max_neighbour_levenstein
        if self.max_neighbour_levenstein_frac is not None:
            max_neighbour_levenstein = int(
                self.max_neighbour_levenstein_frac * len(center_peptide)
            )

        neighbor_peptides = set()

        initial_latent_point = self.encoder_decoder.encode_peptides([center_peptide])[0]
        initial_latent_velocity = torch.zeros_like(initial_latent_point)

        neighborhood = Neigborhood(
            center_peptide=center_peptide,
            trajectories=[],
            # score=self.scored_peptides[center_peptide],
        )

        with parallel_backend("loky"):
            trajectories = Parallel(n_jobs=4)(
                delayed(self.walk_trajectory)(
                    center_peptide,
                    initial_latent_point,
                    initial_latent_velocity,
                    max_neighbour_levenstein,
                )
                for _ in range(self.walker_trajectories_number)
            )

        for trajectory in trajectories:
            # Add the trajectory to the neighborhood
            neighborhood.trajectories.append(trajectory)

            # Collect unique peptides from the trajectory
            for step in trajectory.steps:
                neighbor_peptides.add(step.peptide)

            logger.info(f"Trajectory completed with {len(trajectory.steps)} steps.")

        return neighborhood


class WalkerLocalSearcher(LocalEnumerator):
    def __init__(
        self,
        encoder_decoder: HydrAMPEncoderDecoder,
        walker: RiemannianWalker,
        acceleration_source: AccelerationSource,
        walker_trajectories_number: int,
        walker_time: float,
        max_neighbour_levenstein: int = None,
        max_neighbour_levenstein_frac: float = None,
        device: str = "cpu",
    ):
        super().__init__(max_neighbour_levenstein, max_neighbour_levenstein_frac)
        self.encoder_decoder = encoder_decoder
        self.walker = walker
        self.acceleration_source = acceleration_source
        self.walker_trajectories_number = walker_trajectories_number
        self.walker_time = walker_time
        self.device = device

        self.acceleration_source = acceleration_source
        # TODO: It is quite unclean way to set acceleration_source. It violates Open–closed principle (probably ?)
        self.walker.acceleration_source = acceleration_source

    def reset(self):
        if isinstance(self.acceleration_source, RepulsiveAccelerationSource):
            self.acceleration_source.clear_repulsive_ambient_points()

        self.all_peptides = set()

    def local_enumeration(self, center_peptide, **kwargs) -> Neigborhood:

        # TODO: refactor this part
        if self.max_neighbour_levenstein is None:
            max_neighbour_levenstein = len(center_peptide)
        else:
            max_neighbour_levenstein = self.max_neighbour_levenstein
        if self.max_neighbour_levenstein_frac is not None:
            max_neighbour_levenstein = int(
                self.max_neighbour_levenstein_frac * len(center_peptide)
            )

        neighbor_peptides = set()

        initial_latent_point = self.encoder_decoder.encode_peptides([center_peptide])[0]
        initial_latent_velocity = torch.zeros_like(initial_latent_point)

        neighborhood = Neigborhood(
            center_peptide=center_peptide,
            trajectories=[],
            # score=self.scored_peptides[center_peptide],
        )

        for trajectory_iter in range(self.walker_trajectories_number):
            trajectory = Trajectory(
                steps=[],
                starting_peptide=center_peptide,
                starting_point=initial_latent_point.tolist(),
            )

            self.walker.reset(
                initial_latent_position=initial_latent_point,
                initial_latent_velocity=initial_latent_velocity,
            )

            step_peptide = center_peptide

            walker_step = 0
            while self.walker.time < self.walker_time:
                self.walker.step()

                step = Step.from_riemannian_walker_step(self.walker.current_step)

                current_latent_point = self.walker.current_step.latent_position

                step_peptide = translate_generated_peptide(
                    self.encoder_decoder.decoder_forward(
                        current_latent_point.to(self.device),
                        softmax_and_flatten=False,
                    )
                )
                step.peptide = step_peptide

                neighbor_peptides.add(step_peptide)

                logger.info(
                    f"Trajectory {trajectory_iter} Step {walker_step} Time {self.walker.time} / {self.walker_time} Levenstain {Levenshtein.distance(step_peptide, center_peptide)}: Found {len(neighbor_peptides)} peptides ."
                )
                if (
                    Levenshtein.distance(step_peptide, center_peptide)
                    > max_neighbour_levenstein
                ):
                    logger.info(
                        f"Reached {Levenshtein.distance(step_peptide, center_peptide)} distance. Stopping trajectory."
                    )
                    break

                walker_step += 1
                trajectory.steps.append(step)
            neighborhood.trajectories.append(trajectory)
            logger.info("===========================")

        neighborhood.neighbor_peptides = list(neighbor_peptides)

        return neighborhood


class WalkerMutationLocalSearcherWithAnnealing(LocalEnumerator):
    def __init__(
        self,
        encoder_decoder: HydrAMPEncoderDecoder,
        walker: RiemannianWalker,
        mutation_generator: MutationGenerator,
        acceleration_source: AccelerationSource,
        walker_trajectories_number: int,
        walker_time: float,
        max_neighbour_levenstein: int = None,
        max_neighbour_levenstein_frac: float = None,
        device: str = "cpu",
    ):
        super().__init__(max_neighbour_levenstein, max_neighbour_levenstein_frac)
        self.encoder_decoder = encoder_decoder
        self.walker = walker
        self.mutation_generator = mutation_generator
        self.acceleration_source = acceleration_source
        self.walker_trajectories_number = walker_trajectories_number
        
        self.walker_time = walker_time
        self.max_walker_time = self.walker_time
        self.min_walker_time = self.walker_time / 100
        self.num_steps = self.walker_time / self.walker.spatial_step ** 2
        
        self.device = device

        self.acceleration_source = acceleration_source
        # TODO: It is quite unclean way to set acceleration_source. It violates Open–closed principle (probably ?)
        self.walker.acceleration_source = acceleration_source

    def reset(self):
        if isinstance(self.acceleration_source, RepulsiveAccelerationSource):
            self.acceleration_source.clear_repulsive_ambient_points()

        self.all_peptides = set()

    # @mem_profile
    def local_enumeration(self, center_peptide, current_step=None, max_step=None, **kwargs) -> Neigborhood:

        # TODO: refactor this part
        if self.max_neighbour_levenstein is None:
            max_neighbour_levenstein = len(center_peptide)
        else:
            max_neighbour_levenstein = self.max_neighbour_levenstein
        if self.max_neighbour_levenstein_frac is not None:
            max_neighbour_levenstein = int(
                self.max_neighbour_levenstein_frac * len(center_peptide)
            )

        neighbor_peptides = set()

        initial_latent_point = self.encoder_decoder.encode_peptides([center_peptide])[0]
        initial_latent_velocity = torch.zeros_like(initial_latent_point)

        neighborhood = Neigborhood(
            center_peptide=center_peptide,
            trajectories=[],
            # score=self.scored_peptides[center_peptide],
        )

        if current_step is not None and max_step is not None:

            cosine = 0.5 * (1 + np.cos(np.pi * current_step / max_step))
            self.walker_time = self.min_walker_time + (self.max_walker_time - self.min_walker_time) * cosine

            self.walker.time_step = self.walker_time / self.num_steps
            self.walker.spatial_step = np.sqrt(self.walker.time_step)

        else:
            self.walker_time = self.max_walker_time

        for trajectory_iter in range(self.walker_trajectories_number):
            trajectory = Trajectory(
                steps=[],
                starting_peptide=center_peptide,
                starting_point=initial_latent_point.tolist(),
            )

            self.walker.reset(
                initial_latent_position=initial_latent_point,
                initial_latent_velocity=initial_latent_velocity,
            )

            step_peptide = center_peptide

            walker_step = 0
            while self.walker.time < self.walker_time:

                self.walker.step()

                step = Step.from_riemannian_walker_step(self.walker.current_step)

                mutated_peptides = self.mutation_generator.mutate(step_peptide, self.walker.current_step)

                current_latent_point = self.walker.current_step.latent_position

                step_peptide = translate_generated_peptide(
                    self.encoder_decoder.decoder_forward(
                        current_latent_point.to(self.device),
                        softmax_and_flatten=False,
                    )
                )
                step.peptide = step_peptide

                new_neighbor_peptides = [
                    peptide
                    for peptide in mutated_peptides
                    if Levenshtein.distance(peptide, center_peptide)
                    <= max_neighbour_levenstein
                ]

                # TODO: new_peptides_to_visit is not the best name
                # step.new_peptides_to_visit = new_neighbor_peptides

                neighbor_peptides.update(new_neighbor_peptides)

                new_peptides = [
                    peptide
                    for peptide in mutated_peptides
                    if peptide not in self.all_peptides
                ]

                # step.new_peptides = new_peptides

                if (
                    isinstance(self.acceleration_source, RepulsiveAccelerationSource)
                    and len(new_peptides) != 0
                ):
                    repulsitve_ambient_points = []
                    for peptide in new_peptides:
                        self.all_peptides.add(peptide)
                        repulsitve_ambient_points.append(
                            torch.nn.functional.one_hot(
                                torch.LongTensor(to_one_hot(peptide)),
                                num_classes=21,
                            ).view(-1, 21 * 25)
                        )
                    self.acceleration_source.add_repulsive_ambient_points(
                        torch.cat(repulsitve_ambient_points, dim=0)
                    )

                logger.info(
                    f"Trajectory {trajectory_iter} Step {walker_step} Time {self.walker.time} / {self.walker_time} Levenstain {Levenshtein.distance(step_peptide, center_peptide)}: Found {len(neighbor_peptides)} peptides ."
                )
                if (
                    Levenshtein.distance(step_peptide, center_peptide)
                    > max_neighbour_levenstein
                ):
                    logger.info(
                        f"Reached {Levenshtein.distance(step_peptide, center_peptide)} distance. Stopping trajectory."
                    )
                    break

                walker_step += 1
                trajectory.steps.append(step)
            neighborhood.trajectories.append(trajectory)

        neighborhood.neighbor_peptides = list(neighbor_peptides)

        return neighborhood


class WalkerMutationLocalSearcher(LocalEnumerator):
    def __init__(
        self,
        encoder_decoder: HydrAMPEncoderDecoder,
        walker: RiemannianWalker,
        mutation_generator: MutationGenerator,
        acceleration_source: AccelerationSource,
        walker_trajectories_number: int,
        walker_time: float,
        max_neighbour_levenstein: int = None,
        max_neighbour_levenstein_frac: float = None,
        device: str = "cpu",
    ):
        super().__init__(max_neighbour_levenstein, max_neighbour_levenstein_frac)
        self.encoder_decoder = encoder_decoder
        self.walker = walker
        self.mutation_generator = mutation_generator
        self.acceleration_source = acceleration_source
        self.walker_trajectories_number = walker_trajectories_number
        self.walker_time = walker_time
        self.device = device

        self.acceleration_source = acceleration_source
        # TODO: It is quite unclean way to set acceleration_source. It violates Open–closed principle (probably ?)
        self.walker.acceleration_source = acceleration_source

    def reset(self):
        if isinstance(self.acceleration_source, RepulsiveAccelerationSource):
            self.acceleration_source.clear_repulsive_ambient_points()

        self.all_peptides = set()

    # @mem_profile
    def local_enumeration(self, center_peptide, **kwargs) -> Neigborhood:

        # TODO: refactor this part
        if self.max_neighbour_levenstein is None:
            max_neighbour_levenstein = len(center_peptide)
        else:
            max_neighbour_levenstein = self.max_neighbour_levenstein
        if self.max_neighbour_levenstein_frac is not None:
            max_neighbour_levenstein = int(
                self.max_neighbour_levenstein_frac * len(center_peptide)
            )

        neighbor_peptides = set()

        initial_latent_point = self.encoder_decoder.encode_peptides([center_peptide])[0]
        initial_latent_velocity = torch.zeros_like(initial_latent_point)

        neighborhood = Neigborhood(
            center_peptide=center_peptide,
            trajectories=[],
            # score=self.scored_peptides[center_peptide],
        )

        for trajectory_iter in range(self.walker_trajectories_number):
            trajectory = Trajectory(
                steps=[],
                starting_peptide=center_peptide,
                starting_point=initial_latent_point.tolist(),
            )

            self.walker.reset(
                initial_latent_position=initial_latent_point,
                initial_latent_velocity=initial_latent_velocity,
            )

            step_peptide = center_peptide

            walker_step = 0
            while self.walker.time < self.walker_time:

                self.walker.step()

                step = Step.from_riemannian_walker_step(self.walker.current_step)

                mutated_peptides = self.mutation_generator.mutate(step_peptide, self.walker.current_step)

                current_latent_point = self.walker.current_step.latent_position

                step_peptide = translate_generated_peptide(
                    self.encoder_decoder.decoder_forward(
                        current_latent_point.to(self.device),
                        softmax_and_flatten=False,
                    )
                )
                step.peptide = step_peptide

                new_neighbor_peptides = [
                    peptide
                    for peptide in mutated_peptides
                    if Levenshtein.distance(peptide, center_peptide)
                    <= max_neighbour_levenstein
                ]

                # TODO: new_peptides_to_visit is not the best name
                # step.new_peptides_to_visit = new_neighbor_peptides

                neighbor_peptides.update(new_neighbor_peptides)

                new_peptides = [
                    peptide
                    for peptide in mutated_peptides
                    if peptide not in self.all_peptides
                ]

                # step.new_peptides = new_peptides

                if (
                    isinstance(self.acceleration_source, RepulsiveAccelerationSource)
                    and len(new_peptides) != 0
                ):
                    repulsitve_ambient_points = []
                    for peptide in new_peptides:
                        self.all_peptides.add(peptide)
                        repulsitve_ambient_points.append(
                            torch.nn.functional.one_hot(
                                torch.LongTensor(to_one_hot(peptide)),
                                num_classes=21,
                            ).view(-1, 21 * 25)
                        )
                    self.acceleration_source.add_repulsive_ambient_points(
                        torch.cat(repulsitve_ambient_points, dim=0)
                    )

                logger.info(
                    f"Trajectory {trajectory_iter} Step {walker_step} Time {self.walker.time} / {self.walker_time} Levenstain {Levenshtein.distance(step_peptide, center_peptide)}: Found {len(neighbor_peptides)} peptides ."
                )
                if (
                    Levenshtein.distance(step_peptide, center_peptide)
                    > max_neighbour_levenstein
                ):
                    logger.info(
                        f"Reached {Levenshtein.distance(step_peptide, center_peptide)} distance. Stopping trajectory."
                    )
                    break

                walker_step += 1
                trajectory.steps.append(step)
            neighborhood.trajectories.append(trajectory)

        neighborhood.neighbor_peptides = list(neighbor_peptides)

        return neighborhood


class NormalSamplingLocalSearcher(LocalEnumerator):

    def __init__(
        self,
        encoder_decoder: HydrAMPEncoderDecoder,
        max_neighbour_levenstein: int = None,
        max_neighbour_levenstein_frac: float = None,
        number_of_samples: int = 1000,
        sampling_temperature: float = 1.0,
        batch_size: int = 5000,
        device: str = "cpu",
    ):
        super().__init__(max_neighbour_levenstein, max_neighbour_levenstein_frac)
        self.encoder_decoder = encoder_decoder
        self.device = device

        self.sampling_temperature = sampling_temperature
        self.number_of_samples = number_of_samples

    def local_enumeration(self, center_peptide, batch_size=5000, **kwargs) -> Neigborhood:

        # TODO: refactor this part
        if self.max_neighbour_levenstein is None:
            max_neighbour_levenstein = len(center_peptide)
        else:
            max_neighbour_levenstein = self.max_neighbour_levenstein
        if self.max_neighbour_levenstein_frac is not None:
            max_neighbour_levenstein = int(
                self.max_neighbour_levenstein_frac * len(center_peptide)
            )

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
            sampled_latent_points, batch_size=batch_size
        )

        unique_generated_peptides, indices = np.unique(
            generated_peptides, return_index=True
        )

        filtered_uniqie_generated_peptides = [
            peptide
            for peptide in unique_generated_peptides
            if Levenshtein.distance(peptide, center_peptide) <= max_neighbour_levenstein
        ]

        # trajectory = Trajectory(
        #     steps=[Step(peptide=center_peptide, latent_position=mean.tolist())] + [
        #         Step(
        #             peptide=peptide,
        #             latent_position=sampled_latent_points[index].tolist(),
        #         )
        #         for peptide, index in zip(unique_generated_peptides, indices)
        #         if Levenshtein.distance(peptide, center_peptide)
        #         <= max_neighbour_levenstein
        #     ],
        #     starting_peptide=center_peptide,
        #     starting_point=mean.tolist(),
        # )

        neighborhood = Neigborhood(
            center_peptide=filtered_uniqie_generated_peptides,
            # trajectories=[trajectory],
            neighbor_peptides=list(unique_generated_peptides),
            # score=self.scored_peptides[center_peptide],
        )

        return neighborhood

    
class MutationLocalSearcher(LocalEnumerator):
    def __init__(
        self,
        encoder_decoder: HydrAMPEncoderDecoder,
        mutation_generator: MutationGenerator,
        max_neighbour_levenstein: int = None,
        max_neighbour_levenstein_frac: float = None,
        device: str = "cpu",
    ):
        super().__init__(max_neighbour_levenstein, max_neighbour_levenstein_frac)
        self.encoder_decoder = encoder_decoder
        self.mutation_generator = mutation_generator
        self.device = device
        
    def local_enumeration(self, center_peptide, **kwargs) -> Neigborhood:

        # TODO: refactor this part
        if self.max_neighbour_levenstein is None:
            max_neighbour_levenstein = len(center_peptide)
        else:
            max_neighbour_levenstein = self.max_neighbour_levenstein
        if self.max_neighbour_levenstein_frac is not None:
            max_neighbour_levenstein = int(
                self.max_neighbour_levenstein_frac * len(center_peptide)
            )

        neighbor_peptides = set()

        with torch.no_grad():
            # Encode the center peptide to get the latent point
            center_latent_point = self.encoder_decoder.encode_peptides([center_peptide])[0]
        logger.debug(f"Center latent point device: {center_latent_point.device}")

        neighborhood = Neigborhood(
            center_peptide=center_peptide,
            trajectories=[],
        )

        with torch.no_grad():
            jacobian = self.encoder_decoder.decoder_jacobian(center_latent_point)
            logger.debug(f"jacobian device: {jacobian.device}")
            U, S, V = torch.linalg.svd(jacobian, full_matrices=False)

            logger.debug(f"U, S, V device: {U.device}, {S.device}, {V.device}")
        # TODO: it is stupid that one have to create dummy_step. Refactor it
        dummy_step = RiemannianWalkerStep(latent_position=None, latent_velocity=None)
        dummy_step.U = U
        dummy_step.S = S
        dummy_step.V = V

        mutated_peptides = self.mutation_generator.mutate(center_peptide, dummy_step)
        
        neighbor_peptides = {
            peptide
            for peptide in mutated_peptides
            if Levenshtein.distance(peptide, center_peptide)
            <= max_neighbour_levenstein
        }

        neighborhood.neighbor_peptides = list(neighbor_peptides)

        return neighborhood