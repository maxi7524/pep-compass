"""Run TANDEM pairwise-potential LEBO against the APEX oracle."""

import time
from datetime import datetime

import torch

from pep_compass.local_enumeration.local_enumerator import (
    SamplingFilteredMutationLocalEnumerator,
)
from pep_compass.local_enumeration.mutation.mutation_filters import TandemFilter
from pep_compass.local_enumeration.mutation_enumerator import (
    MutationEnumerationInTangentSpace,
)
from pep_compass.local_enumeration.sampling_walker import (
    SecondOrderRiemannianBrownianEfficientSampling,
)
from pep_compass.models.encoder_decoder.hydramp_encoder_decoder import (
    HydrAMPEncoderDecoder,
)
from pep_compass.optimization.black_box.apex_black_box import APEXBlackBox
from pep_compass.optimization.black_box.csv_observer import CSVObserver
from pep_compass.optimization.lebo.local_enumeration_bayesian_optimizer import (
    LocalEnumerationBayesianOptimizer,
)

DEVICE = "cuda:0"
OUTPUT_PATH = "./results/tandem"
EVALUATION_BUDGET = 1400
TOP_P = 0.9
TEMPERATURE = 1.0
HORIZONTAL_THRESHOLD = 0.1
MAXIMUM_CANDIDATES = 6000
PROTEINS = {
    "middle-1": ("FLYKWWIRIGRLKL", 4),
    "jurand-4": ("KYCRRFRWLTFRWL", 5),
    "jurand-2": ("KFRNRHRWKFKLIFRN", 5),
    "jurand-7": ("KKYWLIRKWIRLWFLT", 5),
    "mammuthusin-3": ("KTLKIIRLLF", 5),
    "hydrodamin-2": ("RMARNLVRYVQGLKKKKVI", 5),
}


def main() -> None:
    black_box = APEXBlackBox(
        mic_aggregate="mean", mic_bacteria=[1, 2, 3], device=DEVICE
    )
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
    walker = SecondOrderRiemannianBrownianEfficientSampling(
        encoder_decoder=encoder_decoder,
        horizontal_threshold=HORIZONTAL_THRESHOLD,
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
    local_enumerator = SamplingFilteredMutationLocalEnumerator(
        encoder_decoder=encoder_decoder,
        sampling_walker=walker,
        mutation_enumerator=mutation_enumerator,
        candidate_filter=TandemFilter(
            encoder_decoder,
            horizontal_threshold=HORIZONTAL_THRESHOLD,
            top_p=TOP_P,
            temperature=TEMPERATURE,
            maximum_candidates=MAXIMUM_CANDIDATES,
        ),
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
    )
    for name, (sequence, repetitions) in PROTEINS.items():
        for repetition in range(repetitions):
            seed = int(time.time()) + repetition
            observer.initialize_observer(
                black_box.get_black_box_info(),
                {
                    "experiment_id": f"tandem_{name}_{seed}_{datetime.now():%Y%m%d_%H%M%S}",
                    "experiment_path": OUTPUT_PATH,
                },
                seed,
                encoder_decoder=encoder_decoder,
            )
            optimizer.optimize(EVALUATION_BUDGET, sequence, seed)


if __name__ == "__main__":
    main()
