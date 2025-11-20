from datetime import datetime
import logging
import time

import torch
from pep_compass.local_enumeration.local_enumerator import SamplingMutationLocalEnumerator
from pep_compass.local_enumeration.mutation_enumerator import MutationEnumerationInTangentSpace
from pep_compass.local_enumeration.sampling_walker import SecondOrderRiemannianBrownianEfficientSampling
from pep_compass.models.encoder_decoder.hydramp_encoder_decoder import HydrAMPEncoderDecoder
from pep_compass.optimization.black_box.apex_black_box import (
    APEXBlackBox,
)
from pep_compass.optimization.black_box.csv_observer import CSVObserver
from pep_compass.optimization.black_box.hydrophobicity_black_box import HydrophobicityBlackBox
from pep_compass.optimization.lebo.local_enumeration_bayesian_optimizer import LocalEnumerationBayesianOptimizer

DEVICE = "cuda:1"
OUTPUT_PATH = "./results/lebo"
EVALUATION_BUDGET = 1400

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s : %(message)s",
    handlers=[
        # logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

# Define the black box and observer
# black_box = APEXBlackBox(
#     mic_aggregate="mean",
#     mic_bacteria=[1, 2, 3],
#     device=DEVICE,
# )

# black_box = BattleAMPBlackBox(device=DEVICE)
# black_box = ToxiPepBlackBox(device=DEVICE)
black_box = HydrophobicityBlackBox(device=DEVICE)

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

# Define sampling walker
sampling_walker = SecondOrderRiemannianBrownianEfficientSampling(
    encoder_decoder=encoder_decoder,
    horizontal_threshold=0.1,
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

# Define local enumerator
local_enumerator = SamplingMutationLocalEnumerator(
    encoder_decoder=encoder_decoder,
    sampling_walker=sampling_walker,
    mutation_enumerator=mutation_enumerator,
    walker_trajectories_number=10,
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
    "middle-1": ("FLYKWWIRIGRLKL", 0),
    "jurand-4": ("KYCRRFRWLTFRWL", 5),
    "jurand-2": ("KFRNRHRWKFKLIFRN", 5),
    "jurand-7": ("KKYWLIRKWIRLWFLT", 5),
    "mammuthusin-3": ("KTLKIIRLLF", 5),
    "hydrodamin-2": ("RMARNLVRYVQGLKKKKVI", 5),
}

for name, (sequence, num) in proteins.items():
    for i in range(num):
        rng_seed = int(time.time())  # Create a unique rng_seed for each iteration
        observer.initialize_observer(
            black_box.get_black_box_info(),
            {"experiment_id": f"{sequence}_{rng_seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}", "experiment_path": OUTPUT_PATH},
            rng_seed,
            encoder_decoder=encoder_decoder,
        )
        optimizer.optimize(evaluation_budget=EVALUATION_BUDGET, starting_point=sequence, rng_seed=rng_seed)
