from datetime import datetime
import logging
import math
import random
import time

import torch
from pep_compass.local_enumeration.local_enumerator import SamplingMutationLocalEnumerator
from pep_compass.local_enumeration.mutation_enumerator import MutationEnumerationInTangentSpace
from pep_compass.local_enumeration.sampling_walker import SecondOrderRiemannianBrownianEfficientSampling
from pep_compass.local_enumeration.mutation.mutation_potentials import (
    ProjectedDirectionPairwiseSimilarityPotential,
    compose_mutant_distribution,
)
from pep_compass.local_enumeration.sampling.sorbes import SubRiemannianTangentSpace
from pep_compass.models.encoder_decoder.hydramp_encoder_decoder import HydrAMPEncoderDecoder
from pep_compass.optimization.black_box.apex_black_box import (
    APEXBlackBox,
)

from pep_compass.optimization.black_box.csv_observer import CSVObserver

from pep_compass.optimization.lebo.local_enumeration_bayesian_optimizer import LocalEnumerationBayesianOptimizer

DEVICE = "cuda:0"
OUTPUT_PATH = "./results/lpbebo_sorbes"
EVALUATION_BUDGET = 1400

# SORBES horizontal/vertical decomposition
SORBES_HORIZONTAL_THRESHOLD = 0.1
# Mutang++ filtering parameters
TOP_P = 0.9  # nucleus filtering threshold for potential-based filtering
TEMPERATURE = 1.0  # mutation score scaling
MAX_CANDIDATES_PER_STEP = 6000

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s : %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

# Define the black box and observer
black_box = APEXBlackBox(
    mic_aggregate="mean",
    mic_bacteria=[1, 2, 3],
    device=DEVICE,
)

observer = CSVObserver(black_box.maximize)
black_box.set_observer(observer)


## Define encoder-decoder
encoder_decoder = HydrAMPEncoderDecoder(
    jacobian_mode="approx",
    device=DEVICE,
    default_condition=torch.tensor([1.0, 1.0]),
    temp=1.0,
    jacobian_eps=0.05,
    field_eps=0.05,
)

# Define sampling walker with SORBES geometry
sampling_walker = SecondOrderRiemannianBrownianEfficientSampling(
    encoder_decoder=encoder_decoder,
    horizontal_threshold=SORBES_HORIZONTAL_THRESHOLD,
    time_step=0.01,
    max_horizontal_update_norm=0.5,
    vertical_movement=True,
)

# Define mutation enumerator
mutation_enumerator = MutationEnumerationInTangentSpace(
    max_len=25,
    direction_significance_threshold=1e-3,
    min_number_of_directions=5,
    token_threshold=0.1,
)

class DynamicSORBESPairwiseSimilarityPotential:
    """Mutation potential using SORBES tangent-space geometry with pairwise similarity scoring."""

    def __init__(self, encoder_decoder, alphabet):
        self.encoder_decoder = encoder_decoder
        self.alphabet = alphabet

    @torch.no_grad()
    def compute(self, parent_peptide: str, mutations: dict[int, list[int]]):
        if not mutations:
            return {}
        z_parent = self.encoder_decoder.encode_peptides([parent_peptide])[0]
        jacobian = self.encoder_decoder.decoder_jacobian(z_parent)
        U, S, V = torch.linalg.svd(jacobian, full_matrices=False)
        # Create tangent space with SORBES geometry
        tangent_space = SubRiemannianTangentSpace(
            U=U,
            S=S,
            V=V,
            horizontal_threshold=SORBES_HORIZONTAL_THRESHOLD,
            device=str(z_parent.device),
        )
        # Use pairwise similarity potential with SORBES geometry
        potential = ProjectedDirectionPairwiseSimilarityPotential(
            tangent_space=tangent_space,
            alphabet=self.alphabet,
        )
        return potential.compute(parent_peptide, mutations)

class SamplingWithMutangPlusPlusLocalEnumerator(SamplingMutationLocalEnumerator):
    """Random walk local enumerator with mutang++ potential-based filtering of candidates."""

    def __init__(
        self,
        encoder_decoder,
        sampling_walker,
        mutation_enumerator,
        potential,
        top_p: float = 0.95,
        temperature: float = 1.0,
        walker_trajectories_number: int = 10,
        time_walk_budget: float = 0.1,
        max_neighbour_levenstein: int = 4,
        device: str = "cpu",
    ):
        super().__init__(
            encoder_decoder=encoder_decoder,
            sampling_walker=sampling_walker,
            mutation_enumerator=mutation_enumerator,
            walker_trajectories_number=walker_trajectories_number,
            time_walk_budget=time_walk_budget,
            max_neighbour_levenstein=max_neighbour_levenstein,
            device=device,
        )
        self.potential = potential
        self.top_p = top_p
        self.temperature = temperature
        self.alphabet = list(" ACDEFGHIKLMNPQRSTVWY")

    @staticmethod
    def _top_p_filter(
        log_potentials,
        top_p: float,
        temperature: float,
    ):
        """Return indices within nucleus of cumulative probability mass."""
        import numpy as np
        if top_p >= 1.0 or len(log_potentials) == 0:
            return np.arange(len(log_potentials))

        scaled = log_potentials / temperature
        scaled -= scaled.max()
        probs = np.exp(scaled)
        probs /= probs.sum()

        cumsum = np.cumsum(probs)
        mask = np.empty(len(cumsum), dtype=bool)
        mask[0] = True
        mask[1:] = cumsum[:-1] < top_p
        return np.nonzero(mask)[0]

    def _cap_mutations(self, mutations: dict[int, list[int]]):
        positions = sorted(mutations.keys())
        if not positions:
            return mutations, 0, 0, 0

        raw_total = math.prod(len(mutations[pos]) for pos in positions)
        if raw_total <= MAX_CANDIDATES_PER_STEP:
            return mutations, raw_total, raw_total, 0

        capped = {pos: list(mutations[pos]) for pos in positions}
        n_pos = len(positions)
        per_pos_cap = max(1, int(MAX_CANDIDATES_PER_STEP ** (1 / n_pos)) - 1)

        for pos in positions:
            if len(capped[pos]) > per_pos_cap:
                capped[pos] = random.sample(capped[pos], per_pos_cap)

        def total_with_parent(muts: dict[int, list[int]]) -> int:
            return math.prod(len(muts[pos]) + 1 for pos in muts)

        capped_total = total_with_parent(capped)
        dropped_positions = 0
        while capped_total > MAX_CANDIDATES_PER_STEP and len(capped) > 1:
            drop_pos = max(capped.keys(), key=lambda p: len(capped[p]))
            capped.pop(drop_pos)
            dropped_positions += 1
            capped_total = total_with_parent(capped)

        return capped, raw_total, capped_total, dropped_positions

    def local_enumeration(self, center_peptide) -> set[str]:
        """Local enumeration with 10 trajectories and mutang++ potential filtering."""
        import Levenshtein
        
        neighbor_peptides = set()
        total_peptides_before_filtering = 0
        total_peptides_after_filtering = 0

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

                new_latent_position, step_info = self.sampling_walker.step(
                    current_latent_position
                )
                adjusted_time_step = step_info["adjusted_time_step"]
                U = step_info["U"].cpu().detach().numpy()
                S = step_info["S"].cpu().detach().numpy()

                mutations = self.mutation_enumerator.get_mutations_from_s_u(S, U)
                if not mutations:
                    logging.info(
                        f"Trajectory {trajectory_iter} Step {walker_step}: No mutation directions found"
                    )
                    break

                mutations, raw_total, capped_total, dropped_positions = self._cap_mutations(
                    mutations
                )
                if raw_total > MAX_CANDIDATES_PER_STEP:
                    logging.info(
                        f"Trajectory {trajectory_iter} Step {walker_step}: Capped mutations from ~{raw_total} to ~{capped_total} combinations"
                        + (f" (dropped {dropped_positions} positions)" if dropped_positions else "")
                    )

                dist = compose_mutant_distribution(
                    parent_peptide=current_peptide,
                    mutations=mutations,
                    potential=self.potential,
                    alphabet=self.alphabet,
                    max_len=self.mutation_enumerator.max_len,
                    include_parent_residue=True,
                )

                sequences = dist.sequences
                log_potentials = dist.log_potentials
                del dist

                if torch.cuda.is_available() and "cuda" in str(self.device):
                    torch.cuda.empty_cache()

                total_peptides_before_filtering += len(sequences)
                logging.info(
                    f"Trajectory {trajectory_iter} Step {walker_step}: Generated {len(sequences)} peptides from mutations"
                )

                if not sequences:
                    logging.info(
                        f"Trajectory {trajectory_iter} Step {walker_step}: No sequences generated, advancing walk"
                    )
                    with torch.no_grad():
                        current_peptide = self.encoder_decoder.decode_peptides(new_latent_position)[0]
                    current_latent_position = new_latent_position
                    time_walk += adjusted_time_step
                    walker_step += 1
                    if Levenshtein.distance(current_peptide, center_peptide) > self.max_neighbour_levenstein:
                        logging.info(
                            f"Reached {Levenshtein.distance(current_peptide, center_peptide)} distance. Stopping trajectory."
                        )
                        break
                    continue

                if len(sequences) > MAX_CANDIDATES_PER_STEP:
                    sampled_idx = random.sample(
                        range(len(sequences)), MAX_CANDIDATES_PER_STEP
                    )
                    sequences = [sequences[i] for i in sampled_idx]
                    log_potentials = log_potentials[sampled_idx]
                    logging.info(
                        f"Trajectory {trajectory_iter} Step {walker_step}: Randomly sampled {len(sequences)} peptides before filtering"
                    )

                kept_idx = self._top_p_filter(
                    log_potentials, self.top_p, self.temperature
                )
                filtered_peptides = [sequences[i] for i in kept_idx]

                total_peptides_after_filtering += len(filtered_peptides)
                retention = len(filtered_peptides) / max(len(sequences), 1)
                logging.info(
                    f"Trajectory {trajectory_iter} Step {walker_step}: After potential filtering: {len(filtered_peptides)} peptides"
                    f" ({100 * retention:.1f}% retention)"
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
                    for peptide in filtered_peptides
                    if Levenshtein.distance(peptide, center_peptide)
                    <= self.max_neighbour_levenstein
                ]

                neighbor_peptides.update(new_neighbor_peptides)

                logging.info(
                    f"Trajectory {trajectory_iter} Step {walker_step} Time {time_walk} / {self.time_walk_budget} Levenshtein {Levenshtein.distance(current_peptide, center_peptide)}: Found {len(neighbor_peptides)} total peptides."
                )
                if (
                    Levenshtein.distance(current_peptide, center_peptide)
                    > self.max_neighbour_levenstein
                ):
                    logging.info(
                        f"Reached {Levenshtein.distance(current_peptide, center_peptide)} distance. Stopping trajectory."
                    )
                    break

        logging.info(
            f"Local enumeration complete: {total_peptides_before_filtering} before filtering, "
            f"{total_peptides_after_filtering} after filtering, {len(neighbor_peptides)} final candidates"
        )
        
        # Store metadata for observer
        self.last_enumeration_stats = {
            "total_before_filtering": total_peptides_before_filtering,
            "total_after_filtering": total_peptides_after_filtering,
            "final_candidates": len(neighbor_peptides)
        }

        return neighbor_peptides


# Define mutation potential with SORBES geometry and pairwise similarity
potential = DynamicSORBESPairwiseSimilarityPotential(
    encoder_decoder=encoder_decoder,
    alphabet=list(" ACDEFGHIKLMNPQRSTVWY"),
)

# Define local enumerator: Random walks with mutang++ potential filtering
local_enumerator = SamplingWithMutangPlusPlusLocalEnumerator(
    encoder_decoder=encoder_decoder,
    sampling_walker=sampling_walker,
    mutation_enumerator=mutation_enumerator,
    potential=potential,
    top_p=TOP_P,
    temperature=TEMPERATURE,
    walker_trajectories_number=10,  # 10 random walk trajectories
    time_walk_budget=0.1,
    max_neighbour_levenstein=4,
    device=DEVICE,
)

#  Define Optimizer
optimizer = LocalEnumerationBayesianOptimizer(
    black_box=black_box,
    device=DEVICE,
    local_enumerator=local_enumerator,
    
    # BO
    initial_peptides_number = 1,
    best_as_center = False,
    acquisition_batch_size = 32,
    standardize = False,
    
    # ROBOT
    levenstain_diversity_threshold = 2,
    evaluations_per_iteration = 3,
    
    # TURBO
    turbo_success_tolerance = 100,
    turbo_failure_tolerance = 1,
    turbo_length_init = 2,
    turbo_length_min = 2,
    turbo_length_max = 2,
    turbo_increase_step = 1,
    turbo_decrease_step = 1,
)

proteins = {
    "middle-1": ("FLYKWWIRIGRLKL", 4),
    "jurand-4": ("KYCRRFRWLTFRWL", 5),
    "jurand-2": ("KFRNRHRWKFKLIFRN", 5),
    "jurand-7": ("KKYWLIRKWIRLWFLT", 5),
    "mammuthusin-3": ("KTLKIIRLLF", 5),
    "hydrodamin-2": ("RMARNLVRYVQGLKKKKVI", 5),
}

for name, (sequence, num) in proteins.items():
    for i in range(num):
        rng_seed = int(time.time())
        observer.initialize_observer(
            black_box.get_black_box_info(),
            {"experiment_id": f"{sequence}_{rng_seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}", "experiment_path": OUTPUT_PATH},
            rng_seed,
            encoder_decoder=encoder_decoder,
        )
        optimizer.optimize(evaluation_budget=EVALUATION_BUDGET, starting_point=sequence, rng_seed=rng_seed)
