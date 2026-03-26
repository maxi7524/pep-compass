import time
import torch
from botorch import fit_fully_bayesian_model_nuts
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.optim import optimize_acqf
from poli_baselines.core.abstract_solver import AbstractBlackBox

from pep_compass.optimization.optimizer import AbstractOptimizer
from pep_compass.utils.utils import set_seed

class SaasboOptimizer(AbstractOptimizer):
    """
    Sparse Axis-Aligned Subspace Bayesian Optimization (SAASBO) implementation.
    
    SAASBO is designed for high-dimensional optimization by learning a sparse 
    axis-aligned subspace and focusing optimization within that subspace.
    
    This implementation minimizes the black_box function with coordinates in [0, 1].
    """

    def __init__(
        self,
        black_box: AbstractBlackBox,
        device: torch.device,
        batch_size: int = 10,
        warmup_steps: int = 128,
        num_samples: int = 64,
        thinning: int = 16,
        dim: int = 64,
    ):
        super().__init__(black_box)

        # Auto-select GPU if available
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tkwargs = {"device": device, "dtype": torch.double}
        self.device = device

        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.num_samples = num_samples
        self.thinning = thinning
        self.dim = dim

    def optimize(
        self, 
        evaluation_budget: int, 
        starting_point: str, 
        rng_seed: int | None = None,
    ):
        if rng_seed is not None:
            set_seed(rng_seed)

        # Encode starting point and move to GPU
        X_encoded = (
            self.black_box.encoder_decoder.encode_peptides([starting_point])
            .to(**self.tkwargs)
        )

        # Scale from [-10, 10] → [0, 1]
        X = (X_encoded + 10) / 20


        # Evaluate black_box (we want to minimize this)
        # Black box expects coordinates in [-10, 10]
        Y = torch.tensor(self.black_box(X_encoded.cpu().detach().numpy())).to(**self.tkwargs)

        print(f"Starting point value: {Y[0].item():.6f}")
        print(f"X range: [{X.min().item():.3f}, {X.max().item():.3f}]")

        n_iterations = evaluation_budget // self.batch_size

        for i in range(n_iterations):
            start_time = time.time()
            train_Y = -Y

            print(f"\nIteration {i + 1}/{n_iterations}")
            print(f"X range: [{X.min().item():.3f}, {X.max().item():.3f}]")
            
            # Fit GP model
            gp = SaasFullyBayesianSingleTaskGP(
                train_X=X,
                train_Y=train_Y.unsqueeze(-1)
                if train_Y.ndim == 1
                else train_Y,
                train_Yvar=torch.full_like(
                    train_Y.unsqueeze(-1)
                    if train_Y.ndim == 1
                    else train_Y,
                    1e-6,
                ),
            ).to(**self.tkwargs)

            # ✅ Fully Bayesian MCMC sampling (runs on GPU if model is CUDA)
            fit_fully_bayesian_model_nuts(
                gp,
                warmup_steps=self.warmup_steps,
                num_samples=self.num_samples,
                thinning=self.thinning,
                disable_progbar=False,
            )
            print("Fitted GP")

            # Acquisition function (on GPU)
            EI = qLogExpectedImprovement(model=gp, best_f=train_Y.max())

            # Search bounds on GPU
            bounds = torch.stack(
                [
                    torch.zeros(self.dim, **self.tkwargs),
                    torch.ones(self.dim, **self.tkwargs),
                ]
            )

            # Optimize acquisition function on GPU
            candidates, acq_values = optimize_acqf(
                EI,
                bounds=bounds,
                q=self.batch_size,
                num_restarts=10,
                raw_samples=1024,
                options={"batch_limit": 5, "maxiter": 100},
            )

            # Convert candidates back to [-10, 10] (still on GPU)
            candidates_original = candidates * 20 - 10

            # Evaluate black_box directly on GPU
            Y_next = torch.tensor(self.black_box(candidates_original.cpu().detach().numpy())).to(**self.tkwargs)

            if Y_next.min() < Y.min():
                ind_best = Y_next.argmin()
                best_candidate = candidates[ind_best]
                print(
                    f"✓ New best: {Y_next[ind_best].item():.6f} @ "
                    f"[{best_candidate[0].item():.4f}, {best_candidate[1].item():.4f}, ...]"
                )
            else:
                print("✗ No improvement this iteration")

            # Update dataset (still all on GPU)
            X = torch.cat([X, candidates])
            Y = torch.cat([Y, Y_next])

            print(f"Iteration took {time.time() - start_time:.2f} seconds")
            print(f"Total evaluations: {len(X)}")
            print(f"Best so far: {Y.min().item():.6f}")
            print(f"Mean of current batch: {Y_next.mean().item():.6f}")

        # Final best
        best_idx = Y.argmin()
        best_X = X[best_idx]
        best_Y = Y[best_idx]

        return {
            "best_x": best_X.detach().cpu().numpy(),
            "best_y": best_Y.item(),
            "all_x": X.detach(),
            "all_y": Y.detach(),
            "n_evaluations": len(X),
        }
