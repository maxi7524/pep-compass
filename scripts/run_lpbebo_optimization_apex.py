from datetime import datetime
import logging
import time

import torch
from pep_compass.local_enumeration.local_enumerator import PotentialFilteredMutationLocalEnumerator
from pep_compass.local_enumeration.mutation.mutation_potentials import DecoderLogProbPotential
from pep_compass.models.encoder_decoder.hydramp_encoder_decoder import HydrAMPEncoderDecoder
from pep_compass.optimization.black_box.apex_black_box import (
    APEXBlackBox,
)

from pep_compass.optimization.black_box.csv_observer import CSVObserver

from pep_compass.optimization.lebo.local_enumeration_bayesian_optimizer import LocalEnumerationBayesianOptimizer

DEVICE = "cuda:0"
OUTPUT_PATH = "./results/lpbebo"
EVALUATION_BUDGET = 1400

TOP_P = 0.9
TEMPERATURE = 1.0

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

# Define mutation potential
potential = DecoderLogProbPotential(encoder_decoder=encoder_decoder)

# Define local enumerator
local_enumerator = PotentialFilteredMutationLocalEnumerator(
    encoder_decoder=encoder_decoder,
    potential=potential,
    top_p=TOP_P,
    temperature=TEMPERATURE,
    direction_significance_threshold=1e-3,
    min_number_of_directions=5,
    token_threshold=0.1,
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
