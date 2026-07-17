"""Random-mutation LE-BO baselines for comparison with the geometry-aware filters (esp. LAMS).

Two control modes (env RAND_MODE), sharing the identical LE-BO outer loop (SORBES walker, ROBOT/TURBO/GP
acquisition, APEX oracle) used by move.py / lebo_plus.py / lpbebo_plus.py:

  RAND_MODE=walker   -- RandomMutationWalker. The SORBES walk still defines the local centre, but the
                        candidate mutations are RANDOM single/multi-position substitutions of the
                        current peptide (random positions x random target residues), NOT read from the
                        decoder Jacobian. This isolates the value of MUTANG's geometry-based
                        enumeration: it is LE-BO with no geometry in the mutation proposal at all.

  RAND_MODE=mutangrandom -- MutangRandomMutation. The MUTANG geometry-based candidate set is enumerated
                        exactly as usual (Jacobian SVD -> tangent mutations -> Cartesian product), but
                        the kept subset is chosen UNIFORMLY AT RANDOM (random nucleus top-p mass)
                        instead of by a similarity filter. This isolates the value of the LAMS/TANDEM/
                        MOVE filter over a random subset of the same MUTANG candidates.

Both keep N_TRAJ trajectories per APEX seed and the same per-trajectory budget, so their per-seed
best-score table is directly comparable to LAMS (lebo_plus.py), TANDEM (lpbebo_plus.py) and MOVE
(move.py). Config via env: RAND_MODE, RAND_N_TRAJ (5), RAND_BUDGET (1400), RAND_TOP_P (0.6, the random
nucleus mass for mutangrandom), RAND_OUT, RAND_DEVICE, RAND_APEX (1 = full 6-seed set).
"""
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

MODE = os.environ.get("RAND_MODE", "walker")            # walker | mutangrandom
assert MODE in ("walker", "mutangrandom"), MODE
DEVICE = os.environ.get("RAND_DEVICE", "cuda:0")
OUTPUT_PATH = os.environ.get("RAND_OUT", f"./results/random_{MODE}")
PEPTIDE = os.environ.get("RAND_PEPTIDE", "FLYKWWIRIGRLKL")
N_TRAJ = int(os.environ.get("RAND_N_TRAJ", "5"))
EVALUATION_BUDGET = int(os.environ.get("RAND_BUDGET", "1400"))
TOP_P = float(os.environ.get("RAND_TOP_P", "0.6"))      # random nucleus mass (mutangrandom)
SORBES_HORIZONTAL_THRESHOLD = 0.1
MAX_CANDIDATES_PER_STEP = 6000
AA = "ACDEFGHIKLMNPQRSTVWY"
# RandomMutationWalker proposal size, matched to MUTANG's typical per-step fan-out:
RW_MAX_POS = int(os.environ.get("RAND_RW_MAX_POS", "5"))   # positions edited per candidate set
RW_RES_PER_POS = int(os.environ.get("RAND_RW_RES_PER_POS", "4"))

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


class _TandemPotential:
    """Only used to build the MUTANG sequence list in mutangrandom mode (its scores are discarded)."""

    def __init__(self, encoder_decoder, alphabet):
        self.encoder_decoder = encoder_decoder
        self.alphabet = alphabet

    @torch.no_grad()
    def compute(self, parent_peptide, mutations):
        if not mutations:
            return {}
        z = self.encoder_decoder.encode_peptides([parent_peptide])[0]
        jac = self.encoder_decoder.decoder_jacobian(z)
        U, S, V = torch.linalg.svd(jac, full_matrices=False)
        ts = SubRiemannianTangentSpace(U=U, S=S, V=V,
                                       horizontal_threshold=SORBES_HORIZONTAL_THRESHOLD,
                                       device=str(z.device))
        return ProjectedDirectionPairwiseSimilarityPotential(
            tangent_space=ts, alphabet=self.alphabet).compute(parent_peptide, mutations)


class RandomLocalEnumerator(SamplingMutationLocalEnumerator):
    def __init__(self, encoder_decoder, sampling_walker, mutation_enumerator, potential,
                 mode="walker", top_p=0.6, walker_trajectories_number=10,
                 time_walk_budget=0.1, max_neighbour_levenstein=4, device="cpu"):
        super().__init__(encoder_decoder=encoder_decoder, sampling_walker=sampling_walker,
                         mutation_enumerator=mutation_enumerator,
                         walker_trajectories_number=walker_trajectories_number,
                         time_walk_budget=time_walk_budget,
                         max_neighbour_levenstein=max_neighbour_levenstein, device=device)
        self.potential = potential
        self.mode = mode
        self.top_p = top_p
        self.alphabet = list(" ACDEFGHIKLMNPQRSTVWY")

    @staticmethod
    def _random_top_p(n, top_p):
        """Indices kept by a uniform-random nucleus of cumulative mass top_p (random subset)."""
        if top_p >= 1.0 or n == 0:
            return np.arange(n)
        probs = np.random.rand(n); probs /= probs.sum()
        order = np.argsort(-probs)
        cumsum = np.cumsum(probs[order])
        mask = np.empty(n, dtype=bool); mask[0] = True; mask[1:] = cumsum[:-1] < top_p
        return order[mask]

    def _cap_mutations(self, mutations):
        """Cap the per-position candidate lists so the Cartesian product stays <= MAX_CANDIDATES
        (the missing step that previously let mutangrandom's product explode and hang)."""
        positions = sorted(mutations.keys())
        if not positions:
            return mutations
        raw = math.prod(len(mutations[p]) for p in positions)
        if raw <= MAX_CANDIDATES_PER_STEP:
            return mutations
        capped = {p: list(mutations[p]) for p in positions}
        per_pos = max(1, int(MAX_CANDIDATES_PER_STEP ** (1 / len(positions))) - 1)
        for p in positions:
            if len(capped[p]) > per_pos:
                capped[p] = random.sample(capped[p], per_pos)

        def tot(m):
            return math.prod(len(m[p]) + 1 for p in m)
        while tot(capped) > MAX_CANDIDATES_PER_STEP and len(capped) > 1:
            capped.pop(max(capped, key=lambda p: len(capped[p])))
        return capped

    def _random_mutations(self, parent):
        """RandomMutationWalker: a MUTANG-shaped mutation set (same fan-out structure) but with
        RANDOM positions and RANDOM target residues -- no Jacobian, no geometry at all."""
        L = len(parent)
        if L == 0:
            return {}
        n_pos = max(1, min(L, RW_MAX_POS))
        positions = random.sample(range(L), n_pos)
        aa_idx = list(range(1, 21))             # the 20 amino acids in self.alphabet (index 0 = pad)
        muts = {}
        for p in positions:
            choices = [i for i in aa_idx if self.alphabet[i] != parent[p]]
            muts[p] = random.sample(choices, min(RW_RES_PER_POS, len(choices)))
        return muts

    def _sequences_from_mutations(self, parent, mutations):
        """Cartesian-product candidate sequences (parent residue retained per position), capped,
        built directly without any potential -- identical candidate set to compose_mutant_distribution
        but ~free, so both controls run at full speed."""
        import itertools
        plist = list(parent); L = len(plist)
        opt = []
        for p in sorted(mutations):
            if p >= L:
                continue
            choices = [plist[p]]
            for idx in mutations[p]:
                ch = self.alphabet[idx]
                if ch != " " and ch != plist[p]:
                    choices.append(ch)
            if len(choices) > 1:
                opt.append((p, choices))
        if not opt:
            return []
        total = math.prod(len(c) for _, c in opt)
        seqs = set()
        if total <= MAX_CANDIDATES_PER_STEP:
            for combo in itertools.product(*[c for _, c in opt]):
                s = plist.copy()
                for (p, _), ch in zip(opt, combo):
                    s[p] = ch
                cand = "".join(s)
                if cand != parent:
                    seqs.add(cand)
        else:
            for _ in range(MAX_CANDIDATES_PER_STEP):
                s = plist.copy()
                for p, c in opt:
                    s[p] = random.choice(c)
                cand = "".join(s)
                if cand != parent:
                    seqs.add(cand)
        return list(seqs)

    def local_enumeration(self, center_peptide) -> set[str]:
        import Levenshtein
        neighbor_peptides = set()
        with torch.no_grad():
            init_latent = self.encoder_decoder.encode_peptides([center_peptide])[0]
        for _ in range(self.walker_trajectories_number):
            current_latent = init_latent
            time_walk = 0.0
            current_peptide = center_peptide
            while time_walk < self.time_walk_budget:
                new_latent, step_info = self.sampling_walker.step(current_latent)
                dt = step_info["adjusted_time_step"]
                if self.mode == "walker":
                    mutations = self._random_mutations(current_peptide)
                else:  # mutangrandom: the real MUTANG mutation set (from this step's SVD)
                    U = step_info["U"].cpu().detach().numpy()
                    S = step_info["S"].cpu().detach().numpy()
                    mutations = self.mutation_enumerator.get_mutations_from_s_u(S, U)
                if mutations:
                    mutations = self._cap_mutations(mutations)
                    sequences = self._sequences_from_mutations(current_peptide, mutations)
                else:
                    sequences = []
                if torch.cuda.is_available() and "cuda" in str(self.device):
                    torch.cuda.empty_cache()
                if sequences and self.mode == "mutangrandom":
                    kept = self._random_top_p(len(sequences), self.top_p)
                    sequences = [sequences[i] for i in kept]
                with torch.no_grad():
                    current_peptide = self.encoder_decoder.decode_peptides(new_latent)[0]
                current_latent = new_latent; time_walk += dt
                neighbor_peptides.update(
                    p for p in sequences
                    if Levenshtein.distance(p, center_peptide) <= self.max_neighbour_levenstein)
                if Levenshtein.distance(current_peptide, center_peptide) > self.max_neighbour_levenstein:
                    break
        return neighbor_peptides


potential = _TandemPotential(encoder_decoder, list(" ACDEFGHIKLMNPQRSTVWY"))
local_enumerator = RandomLocalEnumerator(
    encoder_decoder=encoder_decoder, sampling_walker=sampling_walker,
    mutation_enumerator=mutation_enumerator, potential=potential, mode=MODE, top_p=TOP_P,
    walker_trajectories_number=10, time_walk_budget=0.1, max_neighbour_levenstein=4, device=DEVICE)
optimizer = LocalEnumerationBayesianOptimizer(
    black_box=black_box, device=DEVICE, local_enumerator=local_enumerator,
    initial_peptides_number=1, best_as_center=False, acquisition_batch_size=32,
    standardize=False, levenstain_diversity_threshold=2, evaluations_per_iteration=3,
    turbo_success_tolerance=100, turbo_failure_tolerance=1, turbo_length_init=2,
    turbo_length_min=2, turbo_length_max=2, turbo_increase_step=1, turbo_decrease_step=1)

APEX_PROTEINS = {
    "middle-1": "FLYKWWIRIGRLKL", "jurand-4": "KYCRRFRWLTFRWL",
    "jurand-2": "KFRNRHRWKFKLIFRN", "jurand-7": "KKYWLIRKWIRLWFLT",
    "mammuthusin-3": "KTLKIIRLLF", "hydrodamin-2": "RMARNLVRYVQGLKKKKVI",
}

if __name__ == "__main__":
    if os.environ.get("RAND_APEX", "0") == "1":
        runs = [(n, s) for n, s in APEX_PROTEINS.items() for _ in range(N_TRAJ)]
    else:
        runs = [("single", PEPTIDE)] * N_TRAJ
    print(f"random LE-BO [{MODE}] | {len(runs)} trajectories | top_p={TOP_P} "
          f"| budget={EVALUATION_BUDGET} | device={DEVICE} | out={OUTPUT_PATH}")
    for traj, (name, seq) in enumerate(runs):
        rng_seed = int(time.time()) + traj
        random.seed(rng_seed); np.random.seed(rng_seed % (2**32 - 1))
        observer.initialize_observer(
            black_box.get_black_box_info(),
            {"experiment_id": f"{name}_{seq}_traj{traj}_{rng_seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
             "experiment_path": OUTPUT_PATH},
            rng_seed, encoder_decoder=encoder_decoder)
        t0 = time.time()
        optimizer.optimize(evaluation_budget=EVALUATION_BUDGET, starting_point=seq, rng_seed=rng_seed)
        print(f"[traj {traj+1}/{len(runs)}] {name} done in {time.time()-t0:.1f}s", flush=True)
