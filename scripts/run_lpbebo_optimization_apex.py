from datetime import datetime
import logging
import time

import torch
from pep_compass.local_enumeration.local_enumerator import PotentialFilteredMutationLocalEnumerator
from pep_compass.local_enumeration.mutation.mutation_potentials import (
    ProjectedDirectionPairwiseSimilarityPotential,
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

# Pairwise similarity potential filtering parameters
TOP_P = 0.95  # nucleus filtering threshold
TEMPERATURE = 1.0  # mutation score scaling
SORBES_HORIZONTAL_THRESHOLD = 0.1  # SORBES horizontal/vertical decomposition

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
        # Create tangent space with SORBES geometry (horizontal threshold enables SORBES decomposition)
        tangent_space = SubRiemannianTangentSpace(
            U=U,
            S=S,
            V=V,
            horizontal_threshold=SORBES_HORIZONTAL_THRESHOLD,  # SORBES geometry parameter
            device=str(z_parent.device),
        )
        # Use pairwise similarity potential with SORBES geometry
        potential = ProjectedDirectionPairwiseSimilarityPotential(
            tangent_space=tangent_space,
            alphabet=self.alphabet,
        )
        return potential.compute(parent_peptide, mutations)


# Define mutation potential with SORBES geometry and pairwise similarity
potential = DynamicSORBESPairwiseSimilarityPotential(
    encoder_decoder=encoder_decoder,
    alphabet=list(" ACDEFGHIKLMNPQRSTVWY"),
)

# Define local enumerator: mutang++ with pairwise similarity potential and SORBES geometry
local_enumerator = PotentialFilteredMutationLocalEnumerator(
    encoder_decoder=encoder_decoder,
    potential=potential,
    top_p=TOP_P,  # nucleus filtering - keep only top 95% by cumulative probability
    temperature=TEMPERATURE,
    direction_significance_threshold=1e-3,
    min_number_of_directions=5,
    token_threshold=0.1,  # mutang++ token threshold
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
