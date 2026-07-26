import numpy as np
from poli_baselines.solvers.evolutionary_strategies.cma_es import CMA_ES
import torch

from pep_compass.optimization.optimizer import AbstractOptimizer
from pep_compass.utils.utils import set_seed


class LatentCMAESOptimizer(AbstractOptimizer):
    
    def __init__(
        self,
        black_box,
        device,
    ):
        super().__init__(black_box)
        self.device = device
        self.population_size = 10
        self.initial_sigma = 1.  # Reduced initial sigma to stay closer to valid regions
        self.constraint_penalty = -1000.0
    
    def optimize(
        self, 
        evaluation_budget: int, 
        starting_point, 
        rng_seed: int | None = None,
    ):
        set_seed(rng_seed)
        
        x0 = self.black_box.encoder_decoder.encode_peptides([starting_point]).detach().cpu().numpy()
        y0 = self.black_box(x0)

        max_iterations = evaluation_budget // self.population_size
        # Max - debbuging: Zmiana z wywołania solve(max_iter=0) na zwrot punktu startowego ~CMA-ES uruchamiał całą populację mimo zbyt małego budżetu.
        if max_iterations == 0:
            return {
                'best_x': starting_point,
                'best_y': float(abs(np.asarray(y0).item())),
            }
        
        cma_es_solver = CMA_ES(
            black_box=self.black_box,
            x0=x0,
            y0=y0,
            population_size=self.population_size ,
            initial_mean=x0,
            initial_sigma=self.initial_sigma,
        )
        # print(cma_es_solver)
        solver_result = cma_es_solver.solve(max_iter=max_iterations)
        
        the_best_peptide = self.black_box.encoder_decoder.decode_peptides(torch.tensor(solver_result[0], device=self.device, dtype=torch.float))[0]
        # Max - debbuging: Zmiana z jednoelementowej krotki na float ~przecinek po przypisaniu psuł kontrakt best_y.
        the_best_score = float(abs(np.asarray(solver_result[1]).item()))
        
        result = {
            'best_x': "".join(the_best_peptide),
            'best_y': the_best_score,
        }
        
        print(f"LatentCMAESOptimizer result: {result}")
        return result
