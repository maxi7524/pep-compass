from datetime import datetime
import time
from pep_compass.optimization.baselines.saasbo import SaasboOptimizer
from pep_compass.optimization.black_box.apex_black_box import (
    APEXBlackBox,
    HydrAMPAPEXBlackBox,
)
from pep_compass.optimization.black_box.csv_observer import CSVObserver

DEVICE = "cuda:3"

black_box = HydrAMPAPEXBlackBox(
    mic_aggregate="mean",
    mic_bacteria=[1, 2, 3],
    device=DEVICE,
)
observer = CSVObserver()
black_box.set_observer(observer)

optimizer = SaasboOptimizer(
    black_box=black_box,
    device=DEVICE,
    batch_size=10,
    warmup_steps=256,
    num_samples=128,
    thinning=16,
)

proteins = {
    "middle-1": ("FLYKWWIRIGRLKL", 10),
    "jurand-4": ("KYCRRFRWLTFRWL", 10),
    "jurand-2": ("KFRNRHRWKFKLIFRN", 10),
    "jurand-7": ("KKYWLIRKWIRLWFLT", 10),
    "mammuthusin-3": ("KTLKIIRLLF", 10),
    "hydrodamin-2": ("RMARNLVRYVQGLKKKKVI", 10),
}

for i in range(10):
    for name, (sequence, _) in proteins.items():
        rng_seed = int(time.time())  # Create a unique rng_seed for each iteration
        observer.initialize_observer(
            black_box.get_black_box_info(),
            {
                "experiment_id": f"{sequence}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "experiment_path": "./results",
            },
            rng_seed,
            encoder_decoder=black_box.encoder_decoder,
        )
        optimizer.optimize(
            evaluation_budget=1400, starting_point=sequence, rng_seed=rng_seed
        )
