"""LEBO+ : LE-BO with the **MUTANG+ viability filter** (whitened, Variant-A latent-cosine
similarity used as a hard pairwise threshold) on top of the MUTANG single-position mutation
set, run for the same 6-peptide x 4-5-trajectory APEX benchmark as ``lpbebo_plus.py``.

Concretely, at every step of the random-walk local enumerator:
  1. take the SVD of the decoder Jacobian at the current latent position;
  2. enumerate MUTANG single-position mutations (``MutationEnumerationInTangentSpace``);
  3. form the full Cartesian product of those single-position mutations (the parent residue
     is included so candidates with any subset of mutated positions are considered);
  4. for each multi-mutation candidate, compute the *minimum pairwise whitened cosine
     similarity* between its constituent mutations (see
     ``ProjectedDirectionPairwiseSimilarityPotential._position_vectors``); a candidate with
     only one mutated position has ``min_pair = +inf`` and is always kept;
  5. keep only candidates whose ``min_pair >= LEBO_PLUS_TAU`` (default 0.15) and whose
     Levenshtein distance to the trajectory anchor is within budget.

This is the **product** variant of the MUTANG+ filter from
``analysis/scripts/thesis_figures/_exp_mutangplus_fulldist.py`` and the thesis chapter
``sec:mutang-plus``: filter the full Cartesian product by the minimum pairwise whitened
cosine similarity, with no decoder log-prob weighting and no nucleus selection.

Config via env (mirrors ``lpbebo_plus.py``):
  LEBO_PLUS_PEPTIDE   starting peptide (single-peptide mode)
  LEBO_PLUS_N_TRAJ    number of independent trajectories (single-peptide mode)
  LEBO_PLUS_BUDGET    evaluation budget per trajectory (default 1400; small for smoke)
  LEBO_PLUS_OUT       output dir (default ./results/lebo_plus)
  LEBO_PLUS_DEVICE    cuda:0 | cpu (default cuda:0)
  LEBO_PLUS_TAU       MUTANG+ whitened-similarity threshold (default 0.15)
  LEBO_PLUS_APEX      "1" -> full 6-peptide APEX benchmark (29 trajectories total)
"""
from __future__ import annotations

import logging
import math
import os
import random
import time
from datetime import datetime

import numpy as np
import torch

from pep_compass.local_enumeration.local_enumerator import SamplingMutationLocalEnumerator
from pep_compass.local_enumeration.mutation_enumerator import MutationEnumerationInTangentSpace
from pep_compass.local_enumeration.sampling_walker import (
    SecondOrderRiemannianBrownianEfficientSampling,
)
from pep_compass.local_enumeration.mutation.mutation_potentials import (
    DEFAULT_MAX_LEN,
    MutationPotential,
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


DEVICE = os.environ.get("LEBO_PLUS_DEVICE", "cuda:0")
OUTPUT_PATH = os.environ.get("LEBO_PLUS_OUT", "./results/lebo_plus")
PEPTIDE = os.environ.get("LEBO_PLUS_PEPTIDE", "FLYKWWIRIGRLKL")
N_TRAJ = int(os.environ.get("LEBO_PLUS_N_TRAJ", "10"))
EVALUATION_BUDGET = int(os.environ.get("LEBO_PLUS_BUDGET", "1400"))
TAU = float(os.environ.get("LEBO_PLUS_TAU", "0.15"))

SORBES_HORIZONTAL_THRESHOLD = 0.1
MAX_CANDIDATES_PER_STEP = 6000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s : %(message)s",
    handlers=[logging.StreamHandler()],
)


black_box = APEXBlackBox(mic_aggregate="mean", mic_bacteria=[1, 2, 3], device=DEVICE)
observer = CSVObserver(black_box.maximize)
black_box.set_observer(observer)

encoder_decoder = HydrAMPEncoderDecoder(
    jacobian_mode="approx",
    device=DEVICE,
    default_condition=torch.tensor([1.0, 1.0]),
    temp=1.0,
    jacobian_eps=0.05,
    field_eps=0.05,
)
sampling_walker = SecondOrderRiemannianBrownianEfficientSampling(
    encoder_decoder=encoder_decoder,
    horizontal_threshold=SORBES_HORIZONTAL_THRESHOLD,
    time_step=0.01,
    max_horizontal_update_norm=0.5,
    vertical_movement=True,
)
mutation_enumerator = MutationEnumerationInTangentSpace(
    max_len=25,
    direction_significance_threshold=1e-3,
    min_number_of_directions=5,
    token_threshold=0.1,
)


class _MutangPlusProductPotential(MutationPotential):
    """MUTANG+ product viability potential.

    Wraps ``ProjectedDirectionPairwiseSimilarityPotential`` to obtain the per-position
    whitened-cosine direction matrices, then for each Cartesian-product candidate returns:

        score = min over mutated-mutated position pairs of cos(v_i, v_j)

    with sensible neutral defaults:
      * the all-parent candidate is excluded (no mutations at all),
      * a candidate with exactly one mutated position has ``score = +inf`` (no pairs to
        score, always passes the downstream tau filter),
      * candidates with several mutations get the minimum pairwise whitened cosine
        between their constituent mutated single-position directions.

    The downstream pipeline filters with ``score >= tau`` (hard MUTANG+ product filter).
    """

    def __init__(self, base: ProjectedDirectionPairwiseSimilarityPotential):
        self._base = base
        self.alphabet = base.alphabet

    @torch.no_grad()
    def compute(
        self, parent_peptide: str, mutations: dict[int, list[int]]
    ) -> dict[tuple[int, ...], float]:
        positions = sorted(mutations.keys())
        if not positions:
            return {}

        device = self._base.tangent_space.device
        padded = parent_peptide.ljust(DEFAULT_MAX_LEN)
        parent_aa_indices = torch.tensor(
            [self._base.alphabet.index(padded[pos]) for pos in positions],
            device=device,
            dtype=torch.long,
        )

        vectors_per_position = []
        aa_indices_per_position = []
        for i_pos, pos in enumerate(positions):
            aa_indices = torch.tensor(mutations[pos], device=device, dtype=torch.long)
            aa_indices_per_position.append(aa_indices)
            parent_aa_idx = int(parent_aa_indices[i_pos].item())
            vectors_per_position.append(
                self._base._position_vectors(pos, aa_indices, parent_aa_idx)
            )

        n_pos = len(positions)
        index_ranges = [
            torch.arange(len(a), device=device) for a in aa_indices_per_position
        ]
        if n_pos == 1:
            combo_indices = index_ranges[0].unsqueeze(1)
        else:
            combo_indices = torch.cartesian_prod(*index_ranges)

        aa_choice = torch.stack(
            [
                aa_indices_per_position[i][combo_indices[:, i]]
                for i in range(n_pos)
            ],
            dim=1,
        )
        mut_mask = aa_choice != parent_aa_indices
        has_mutation = mut_mask.any(dim=1)

        if n_pos < 2:
            min_pair = torch.full((combo_indices.shape[0],), float("inf"), device=device)
        else:
            selected_vectors = torch.stack(
                [
                    vectors_per_position[i][combo_indices[:, i]]
                    for i in range(n_pos)
                ],
                dim=1,
            )
            cos_matrix = selected_vectors @ selected_vectors.transpose(1, 2)
            i_idx, j_idx = torch.triu_indices(n_pos, n_pos, offset=1, device=device)
            pairwise_cos_all = cos_matrix[:, i_idx, j_idx]
            mut_i = mut_mask[:, i_idx]
            mut_j = mut_mask[:, j_idx]
            both_mutated_mask = mut_i & mut_j
            big = torch.full_like(pairwise_cos_all, float("inf"))
            masked_cos = torch.where(both_mutated_mask, pairwise_cos_all, big)
            any_pair = both_mutated_mask.any(dim=1)
            min_pair = masked_cos.min(dim=1).values
            min_pair = torch.where(any_pair, min_pair, torch.full_like(min_pair, float("inf")))

        valid_mask = has_mutation
        aa_choice_valid = aa_choice[valid_mask].cpu().tolist()
        score_valid = min_pair[valid_mask].cpu().tolist()
        return {
            tuple(aa_tuple): float(score)
            for aa_tuple, score in zip(aa_choice_valid, score_valid)
        }


class DynamicSORBESMutangPlusPotential:
    """Build the MUTANG+ product viability potential from the parent's tangent space."""

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
            U=U,
            S=S,
            V=V,
            horizontal_threshold=SORBES_HORIZONTAL_THRESHOLD,
            device=str(z_parent.device),
        )
        base = ProjectedDirectionPairwiseSimilarityPotential(
            tangent_space=tangent_space, alphabet=self.alphabet,
        )
        return _MutangPlusProductPotential(base).compute(parent_peptide, mutations)


class SamplingWithMutangPlusLocalEnumerator(SamplingMutationLocalEnumerator):
    """Random-walk enumerator with the MUTANG+ hard-threshold product filter."""

    def __init__(
        self,
        encoder_decoder,
        sampling_walker,
        mutation_enumerator,
        potential,
        tau: float = 0.15,
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
        self.tau = tau
        self.alphabet = list(" ACDEFGHIKLMNPQRSTVWY")

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

        neighbor_peptides: set[str] = set()
        with torch.no_grad():
            initial_latent_position = self.encoder_decoder.encode_peptides([center_peptide])[0]

        for _ in range(self.walker_trajectories_number):
            current_latent_position = initial_latent_position
            time_walk = 0.0
            current_peptide = center_peptide
            while time_walk < self.time_walk_budget:
                new_latent_position, step_info = self.sampling_walker.step(
                    current_latent_position
                )
                adjusted_time_step = step_info["adjusted_time_step"]
                U = step_info["U"].cpu().detach().numpy()
                S = step_info["S"].cpu().detach().numpy()
                mutations = self.mutation_enumerator.get_mutations_from_s_u(S, U)
                if not mutations:
                    break
                mutations, _, _, _ = self._cap_mutations(mutations)
                dist = compose_mutant_distribution(
                    parent_peptide=current_peptide,
                    mutations=mutations,
                    potential=self.potential,
                    alphabet=self.alphabet,
                    max_len=self.mutation_enumerator.max_len,
                    include_parent_residue=True,
                )
                sequences = dist.sequences
                scores = dist.log_potentials
                del dist
                if torch.cuda.is_available() and "cuda" in str(self.device):
                    torch.cuda.empty_cache()

                if not sequences:
                    with torch.no_grad():
                        current_peptide = self.encoder_decoder.decode_peptides(
                            new_latent_position
                        )[0]
                    current_latent_position = new_latent_position
                    time_walk += adjusted_time_step
                    if Levenshtein.distance(current_peptide, center_peptide) > self.max_neighbour_levenstein:
                        break
                    continue

                if len(sequences) > MAX_CANDIDATES_PER_STEP:
                    idx = random.sample(range(len(sequences)), MAX_CANDIDATES_PER_STEP)
                    sequences = [sequences[i] for i in idx]
                    scores = scores[idx]

                kept_idx = np.where(scores >= self.tau)[0]
                filtered = [sequences[i] for i in kept_idx]
                with torch.no_grad():
                    current_peptide = self.encoder_decoder.decode_peptides(new_latent_position)[0]
                current_latent_position = new_latent_position
                time_walk += adjusted_time_step
                neighbor_peptides.update(
                    p
                    for p in filtered
                    if Levenshtein.distance(p, center_peptide) <= self.max_neighbour_levenstein
                )
                if (
                    Levenshtein.distance(current_peptide, center_peptide)
                    > self.max_neighbour_levenstein
                ):
                    break
        return neighbor_peptides


potential = DynamicSORBESMutangPlusPotential(
    encoder_decoder=encoder_decoder,
    alphabet=list(" ACDEFGHIKLMNPQRSTVWY"),
)
local_enumerator = SamplingWithMutangPlusLocalEnumerator(
    encoder_decoder=encoder_decoder,
    sampling_walker=sampling_walker,
    mutation_enumerator=mutation_enumerator,
    potential=potential,
    tau=TAU,
    walker_trajectories_number=10,
    time_walk_budget=0.1,
    max_neighbour_levenstein=4,
    device=DEVICE,
)
optimizer = LocalEnumerationBayesianOptimizer(
    black_box=black_box,
    device=DEVICE,
    local_enumerator=local_enumerator,
    initial_peptides_number=1,
    best_as_center=False,
    acquisition_batch_size=32,
    standardize=False,
    levenstain_diversity_threshold=2,
    evaluations_per_iteration=3,
    turbo_success_tolerance=100,
    turbo_failure_tolerance=1,
    turbo_length_init=2,
    turbo_length_min=2,
    turbo_length_max=2,
    turbo_increase_step=1,
    turbo_decrease_step=1,
)


APEX_PROTEINS = {
    "middle-1": ("FLYKWWIRIGRLKL", 4),
    "jurand-4": ("KYCRRFRWLTFRWL", 5),
    "jurand-2": ("KFRNRHRWKFKLIFRN", 5),
    "jurand-7": ("KKYWLIRKWIRLWFLT", 5),
    "mammuthusin-3": ("KTLKIIRLLF", 5),
    "hydrodamin-2": ("RMARNLVRYVQGLKKKKVI", 5),
}


if __name__ == "__main__":
    if os.environ.get("LEBO_PLUS_APEX", "0") == "1":
        runs = [(name, seq) for name, (seq, num) in APEX_PROTEINS.items() for _ in range(num)]
    else:
        runs = [("single", PEPTIDE)] * N_TRAJ
    print(
        f"LEBO+ (MUTANG+ product, tau={TAU}) | {len(runs)} trajectories "
        f"| budget={EVALUATION_BUDGET} | device={DEVICE} | out={OUTPUT_PATH}",
        flush=True,
    )
    for traj, (name, seq) in enumerate(runs):
        rng_seed = int(time.time()) + traj
        observer.initialize_observer(
            black_box.get_black_box_info(),
            {
                "experiment_id": (
                    f"{name}_{seq}_traj{traj}_{rng_seed}_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                ),
                "experiment_path": OUTPUT_PATH,
            },
            rng_seed,
            encoder_decoder=encoder_decoder,
        )
        t0 = time.time()
        optimizer.optimize(
            evaluation_budget=EVALUATION_BUDGET,
            starting_point=seq,
            rng_seed=rng_seed,
        )
        print(
            f"[traj {traj+1}/{len(runs)}] {name} done in {time.time()-t0:.1f}s",
            flush=True,
        )
