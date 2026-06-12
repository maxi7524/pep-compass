"""LPBEBO+ : LEBO/LPBEBO with the MUTANG++ (TANDEM) pairwise-similarity potential as the
candidate filter, run for a configurable number of trajectories so a best-score table (à la
PepCompass LEBO + baselines) can be built afterwards from the per-evaluation CSVs.

MUTANG++ is the old name for TANDEM; the candidate enumerator filters the random-walk neighbours
by the projected-direction pairwise-similarity potential, exactly as in
``run_lpbebo_optimization_apex.py``. This script only parameterises that pipeline (peptide, number
of trajectories, evaluation budget) and runs N independent trajectories with different seeds.

Config via env:
  LPBEBO_PEPTIDE   starting peptide (default middle-1 FLYKWWIRIGRLKL)
  LPBEBO_N_TRAJ    number of independent trajectories (default 10)
  LPBEBO_BUDGET    evaluation budget per trajectory (default 1400; use a small value for smoke)
  LPBEBO_OUT       output dir for the CSVObserver (default ./results/lpbebo_plus)
  LPBEBO_DEVICE    cuda:0 | cpu  (default cuda:0)
"""
import logging
import math
import os
import random
import time
from datetime import datetime

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
from pep_compass.optimization.black_box.apex_black_box import APEXBlackBox
from pep_compass.optimization.black_box.csv_observer import CSVObserver
from pep_compass.optimization.lebo.local_enumeration_bayesian_optimizer import (
    LocalEnumerationBayesianOptimizer,
)

DEVICE = os.environ.get("LPBEBO_DEVICE", "cuda:0")
OUTPUT_PATH = os.environ.get("LPBEBO_OUT", "./results/lpbebo_plus")
PEPTIDE = os.environ.get("LPBEBO_PEPTIDE", "FLYKWWIRIGRLKL")
N_TRAJ = int(os.environ.get("LPBEBO_N_TRAJ", "10"))
EVALUATION_BUDGET = int(os.environ.get("LPBEBO_BUDGET", "1400"))

SORBES_HORIZONTAL_THRESHOLD = 0.1
TOP_P = float(os.environ.get("LPBEBO_TOP_P", "0.9"))  # nucleus (cumulative prob-mass) filter
TEMPERATURE = 1.0
MAX_CANDIDATES_PER_STEP = 6000

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s : %(message)s",
                    handlers=[logging.StreamHandler()])

black_box = APEXBlackBox(mic_aggregate="mean", mic_bacteria=[1, 2, 3], device=DEVICE)
observer = CSVObserver(black_box.maximize)
black_box.set_observer(observer)

encoder_decoder = HydrAMPEncoderDecoder(
    jacobian_mode="approx", device=DEVICE,
    default_condition=torch.tensor([1.0, 1.0]), temp=1.0,
    jacobian_eps=0.05, field_eps=0.05,
)
sampling_walker = SecondOrderRiemannianBrownianEfficientSampling(
    encoder_decoder=encoder_decoder, horizontal_threshold=SORBES_HORIZONTAL_THRESHOLD,
    time_step=0.01, max_horizontal_update_norm=0.5, vertical_movement=True,
)
mutation_enumerator = MutationEnumerationInTangentSpace(
    max_len=25, direction_significance_threshold=1e-3,
    min_number_of_directions=5, token_threshold=0.1,
)


class DynamicSORBESPairwiseSimilarityPotential:
    """MUTANG++ (TANDEM) potential: pairwise-similarity scoring on the SORBES tangent space."""

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
        tangent_space = SubRiemannianTangentSpace(
            U=U, S=S, V=V, horizontal_threshold=SORBES_HORIZONTAL_THRESHOLD,
            device=str(z_parent.device),
        )
        potential = ProjectedDirectionPairwiseSimilarityPotential(
            tangent_space=tangent_space, alphabet=self.alphabet,
        )
        return potential.compute(parent_peptide, mutations)


class SamplingWithMutangPlusPlusLocalEnumerator(SamplingMutationLocalEnumerator):
    """Random-walk local enumerator with MUTANG++ (TANDEM) potential-based candidate filtering."""

    def __init__(self, encoder_decoder, sampling_walker, mutation_enumerator, potential,
                 top_p=0.95, temperature=1.0, walker_trajectories_number=10,
                 time_walk_budget=0.1, max_neighbour_levenstein=4, device="cpu"):
        super().__init__(
            encoder_decoder=encoder_decoder, sampling_walker=sampling_walker,
            mutation_enumerator=mutation_enumerator,
            walker_trajectories_number=walker_trajectories_number,
            time_walk_budget=time_walk_budget,
            max_neighbour_levenstein=max_neighbour_levenstein, device=device,
        )
        self.potential = potential
        self.top_p = top_p
        self.temperature = temperature
        self.alphabet = list(" ACDEFGHIKLMNPQRSTVWY")

    @staticmethod
    def _top_p_filter(log_potentials, top_p, temperature):
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

    def _cap_mutations(self, mutations):
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

        def total_with_parent(muts):
            return math.prod(len(muts[pos]) + 1 for pos in muts)

        capped_total = total_with_parent(capped)
        dropped = 0
        while capped_total > MAX_CANDIDATES_PER_STEP and len(capped) > 1:
            drop_pos = max(capped.keys(), key=lambda p: len(capped[p]))
            capped.pop(drop_pos)
            dropped += 1
            capped_total = total_with_parent(capped)
        return capped, raw_total, capped_total, dropped

    def local_enumeration(self, center_peptide) -> set[str]:
        import Levenshtein
        neighbor_peptides = set()
        with torch.no_grad():
            initial_latent_position = self.encoder_decoder.encode_peptides([center_peptide])[0]

        for trajectory_iter in range(self.walker_trajectories_number):
            current_latent_position = initial_latent_position
            time_walk = 0.0
            current_peptide = center_peptide
            walker_step = 0
            while time_walk < self.time_walk_budget:
                new_latent_position, step_info = self.sampling_walker.step(current_latent_position)
                adjusted_time_step = step_info["adjusted_time_step"]
                U = step_info["U"].cpu().detach().numpy()
                S = step_info["S"].cpu().detach().numpy()
                mutations = self.mutation_enumerator.get_mutations_from_s_u(S, U)
                if not mutations:
                    break
                mutations, raw_total, capped_total, _ = self._cap_mutations(mutations)
                dist = compose_mutant_distribution(
                    parent_peptide=current_peptide, mutations=mutations,
                    potential=self.potential, alphabet=self.alphabet,
                    max_len=self.mutation_enumerator.max_len, include_parent_residue=True,
                )
                sequences = dist.sequences
                log_potentials = dist.log_potentials
                del dist
                if torch.cuda.is_available() and "cuda" in str(self.device):
                    torch.cuda.empty_cache()
                if not sequences:
                    with torch.no_grad():
                        current_peptide = self.encoder_decoder.decode_peptides(new_latent_position)[0]
                    current_latent_position = new_latent_position
                    time_walk += adjusted_time_step
                    walker_step += 1
                    if Levenshtein.distance(current_peptide, center_peptide) > self.max_neighbour_levenstein:
                        break
                    continue
                if len(sequences) > MAX_CANDIDATES_PER_STEP:
                    idx = random.sample(range(len(sequences)), MAX_CANDIDATES_PER_STEP)
                    sequences = [sequences[i] for i in idx]
                    log_potentials = log_potentials[idx]
                kept_idx = self._top_p_filter(log_potentials, self.top_p, self.temperature)
                filtered = [sequences[i] for i in kept_idx]
                with torch.no_grad():
                    current_peptide = self.encoder_decoder.decode_peptides(new_latent_position)[0]
                current_latent_position = new_latent_position
                time_walk += adjusted_time_step
                walker_step += 1
                neighbor_peptides.update(
                    p for p in filtered
                    if Levenshtein.distance(p, center_peptide) <= self.max_neighbour_levenstein
                )
                if Levenshtein.distance(current_peptide, center_peptide) > self.max_neighbour_levenstein:
                    break
        return neighbor_peptides


potential = DynamicSORBESPairwiseSimilarityPotential(
    encoder_decoder=encoder_decoder, alphabet=list(" ACDEFGHIKLMNPQRSTVWY"),
)
local_enumerator = SamplingWithMutangPlusPlusLocalEnumerator(
    encoder_decoder=encoder_decoder, sampling_walker=sampling_walker,
    mutation_enumerator=mutation_enumerator, potential=potential,
    top_p=TOP_P, temperature=TEMPERATURE, walker_trajectories_number=10,
    time_walk_budget=0.1, max_neighbour_levenstein=4, device=DEVICE,
)
optimizer = LocalEnumerationBayesianOptimizer(
    black_box=black_box, device=DEVICE, local_enumerator=local_enumerator,
    initial_peptides_number=1, best_as_center=False, acquisition_batch_size=32,
    standardize=False, levenstain_diversity_threshold=2, evaluations_per_iteration=3,
    turbo_success_tolerance=100, turbo_failure_tolerance=1, turbo_length_init=2,
    turbo_length_min=2, turbo_length_max=2, turbo_increase_step=1, turbo_decrease_step=1,
)

# Full APEX benchmark set (6 peptides x 4-5 seeds = 29 runs), matching the PepCompass-style
# LEBO + baselines comparison. Enabled with LPBEBO_APEX=1.
APEX_PROTEINS = {
    "middle-1": ("FLYKWWIRIGRLKL", 4),
    "jurand-4": ("KYCRRFRWLTFRWL", 5),
    "jurand-2": ("KFRNRHRWKFKLIFRN", 5),
    "jurand-7": ("KKYWLIRKWIRLWFLT", 5),
    "mammuthusin-3": ("KTLKIIRLLF", 5),
    "hydrodamin-2": ("RMARNLVRYVQGLKKKKVI", 5),
}

if __name__ == "__main__":
    if os.environ.get("LPBEBO_APEX", "0") == "1":
        # N_TRAJ independent trajectories PER peptide (LPBEBO_N_TRAJ), not the per-peptide seed counts.
        runs = [(name, seq) for name, (seq, _num) in APEX_PROTEINS.items() for _ in range(N_TRAJ)]
    else:
        runs = [("single", PEPTIDE)] * N_TRAJ
    print(f"LPBEBO+ (MUTANG++/TANDEM filter) | {len(runs)} trajectories "
          f"| top_p(mass)={TOP_P} | budget={EVALUATION_BUDGET} | device={DEVICE} | out={OUTPUT_PATH}")
    for traj, (name, seq) in enumerate(runs):
        rng_seed = int(time.time()) + traj
        observer.initialize_observer(
            black_box.get_black_box_info(),
            {"experiment_id": f"{name}_{seq}_traj{traj}_{rng_seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
             "experiment_path": OUTPUT_PATH},
            rng_seed, encoder_decoder=encoder_decoder,
        )
        t0 = time.time()
        optimizer.optimize(evaluation_budget=EVALUATION_BUDGET, starting_point=seq, rng_seed=rng_seed)
        print(f"[traj {traj+1}/{len(runs)}] {name} done in {time.time()-t0:.1f}s", flush=True)
