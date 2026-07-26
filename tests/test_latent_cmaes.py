"""Regression tests for the latent CMA-ES optimizer."""

import numpy as np
import torch

from pep_compass.optimization.baselines import latent_cmaes


class _EncoderDecoder:
    def encode_peptides(self, peptides: list[str]) -> torch.Tensor:
        return torch.zeros((len(peptides), 64))

    def decode_peptides(self, latent: torch.Tensor) -> list[list[str]]:
        return [list("PEPTIDE")]


class _BlackBox:
    def __init__(self) -> None:
        self.encoder_decoder = _EncoderDecoder()
        self.calls = 0

    def __call__(self, latent: np.ndarray) -> np.ndarray:
        self.calls += len(latent)
        return np.zeros((len(latent), 1))


class _Solver:
    def __init__(self, **kwargs: object) -> None:
        pass

    def solve(self, max_iter: int) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(64), np.array([-2.5])


def test_optimize_returns_scalar_best_score(monkeypatch) -> None:
    """Expose ``best_y`` as a float instead of a singleton tuple."""
    monkeypatch.setattr(latent_cmaes, "CMA_ES", _Solver)
    black_box = _BlackBox()
    optimizer = latent_cmaes.LatentCMAESOptimizer(black_box, device="cpu")

    result = optimizer.optimize(10, "PEPTIDE", rng_seed=123)

    assert result["best_y"] == 2.5
    assert isinstance(result["best_y"], float)


def test_optimize_does_not_start_population_above_budget(monkeypatch) -> None:
    """Return the starting point when no CMA-ES generation fits the budget."""

    def fail_if_constructed(**kwargs: object) -> None:
        raise AssertionError("CMA-ES solver must not be constructed")

    monkeypatch.setattr(latent_cmaes, "CMA_ES", fail_if_constructed)
    black_box = _BlackBox()
    optimizer = latent_cmaes.LatentCMAESOptimizer(black_box, device="cpu")

    result = optimizer.optimize(1, "PEPTIDE", rng_seed=123)

    assert result == {"best_x": "PEPTIDE", "best_y": 0.0}
    assert black_box.calls == 1
