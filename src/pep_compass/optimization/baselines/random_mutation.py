import numpy as np
from poli_baselines.solvers.simple.random_mutation import RandomMutation
from poli.core.abstract_black_box import AbstractBlackBox



from pep_compass.optimization.black_box.negative_black_box import NegativeBlackBox
from pep_compass.optimization.optimizer import AbstractOptimizer
from pep_compass.models.esm import ESM2PPLScorer
from pep_compass.utils.utils import set_seed


class ESMFilteredRandomMutation(RandomMutation):
    def __init__(
        self,
        *args,
        esm_scorer: ESM2PPLScorer,
        esm_ppl_threshold: float,
        esm_max_resampling_attempts: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.esm_scorer = esm_scorer
        self.esm_ppl_threshold = esm_ppl_threshold
        self.esm_max_resampling_attempts = esm_max_resampling_attempts

    def next_candidate(self) -> np.ndarray:
        candidates: list[np.ndarray] = []
        for _ in range(self.batch_size):
            candidates.append(self._sample_passing_candidate().reshape(1, -1))
        return np.concatenate(candidates, axis=0)

    def _sample_passing_candidate(self) -> np.ndarray:
        for _ in range(self.esm_max_resampling_attempts):
            candidate = self._next_candidate()
            sequence = "".join(candidate.reshape(-1))
            if self.esm_scorer.passes_threshold(sequence, self.esm_ppl_threshold):
                return candidate
        raise RuntimeError(
            "Could not find a mutation that passes ESM PPL threshold "
            f"{self.esm_ppl_threshold} after {self.esm_max_resampling_attempts} attempts."
        )


class RandomMutationOptimizer(AbstractOptimizer):
    
    def __init__(
        self,
        black_box: AbstractBlackBox,
        esm_model_name: str | None = None,
        esm_ppl_threshold: float = -0.5,
        esm_device: str = "cpu",
        esm_max_resampling_attempts: int = 200,
    ):
        if not black_box.maximize:
            black_box = NegativeBlackBox(black_box)

        super().__init__(black_box)
        self.esm_model_name = esm_model_name
        self.esm_ppl_threshold = esm_ppl_threshold
        self.esm_device = esm_device
        self.esm_max_resampling_attempts = esm_max_resampling_attempts
        self.esm_scorer = (
            ESM2PPLScorer(model_name=esm_model_name, device=esm_device)
            if esm_model_name is not None
            else None
        )
    
    def optimize(
        self, 
        evaluation_budget: int, 
        starting_point, 
        rng_seed: int | None = None,
    ):
        set_seed(rng_seed)

        if self.esm_scorer is None:
            random_mutation_solver = RandomMutation(
                black_box=self.black_box,
                x0=np.array([starting_point]),
                y0=self.black_box(np.array([starting_point])),
            )
            solver_result = random_mutation_solver.solve(max_iter=evaluation_budget)
            result = {
                'best_x': "".join(solver_result[0][0]),
                'best_y': abs(solver_result[1].item()),
            }
            print(f"RandomMutationOptimizer result: {result}")
            return result

        starting_sequence = str(starting_point)
        starting_pll = self.esm_scorer.pll(starting_sequence)
        if starting_pll < self.esm_ppl_threshold:
            raise ValueError(
                f"Starting sequence '{starting_sequence}' has ESM PLL {starting_pll:.4f}, "
                f"below threshold {self.esm_ppl_threshold}."
            )

        random_mutation_solver = ESMFilteredRandomMutation(
            black_box=self.black_box,
            x0=np.array([starting_sequence]),
            y0=self.black_box(np.array([starting_sequence])),
            esm_scorer=self.esm_scorer,
            esm_ppl_threshold=self.esm_ppl_threshold,
            esm_max_resampling_attempts=self.esm_max_resampling_attempts,
        )
        solver_result = random_mutation_solver.solve(max_iter=evaluation_budget)
        result = {
            'best_x': "".join(solver_result[0][0]),
            'best_y': abs(solver_result[1].item()),
        }
        
        print(f"RandomMutationOptimizer result: {result}")
        return result
