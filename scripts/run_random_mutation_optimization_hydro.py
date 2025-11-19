from datetime import datetime
import time
from pep_compass.optimization.baselines.random_mutation import RandomMutationOptimizer
from pep_compass.optimization.black_box.battleamp_black_box import BattleAMPBlackBox
from pep_compass.optimization.black_box.csv_observer import CSVObserver
from pep_compass.optimization.black_box.toxipep_black_box import ToxiPepBlackBox

DEVICE = "cuda"

# black_box = APEXBlackBox(
#     mic_aggregate="mean",
#     mic_bacteria=[1, 2, 3],
#     device=DEVICE,
# )

black_box = BattleAMPBlackBox(device=DEVICE)

# black_box = ToxiPepBlackBox(device=DEVICE)

observer = CSVObserver()
black_box.set_observer(observer)

optimizer = RandomMutationOptimizer(
    black_box=black_box,
)

proteins = {
    "middle-1": ("FLYKWWIRIGRLKL", 5),
    "jurand-4": ("KYCRRFRWLTFRWL", 5),
    "jurand-2": ("KFRNRHRWKFKLIFRN", 5),
    "jurand-7": ("KKYWLIRKWIRLWFLT", 5),
    "mammuthusin-3": ("KTLKIIRLLF", 5),
    "hydrodamin-2": ("RMARNLVRYVQGLKKKKVI", 5),
}

for i in range(5):
    for name, (sequence, _) in proteins.items():
        print(f"Starting optimization for {name}, iteration {i+1}")
        rng_seed = int(time.time())  # Create a unique rng_seed for each iteration
        observer.initialize_observer(
            black_box.get_black_box_info(),
            {
                "experiment_id": f"{sequence}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "experiment_path": "./results/random_mutation",
            },
            rng_seed,
            
        )
        optimizer.optimize(
            evaluation_budget=1400, starting_point=sequence, rng_seed=rng_seed
        )

