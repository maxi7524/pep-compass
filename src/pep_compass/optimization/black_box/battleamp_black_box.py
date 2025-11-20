import numpy as np
import torch
from poli_baselines.core.abstract_solver import AbstractBlackBox
from poli.core.black_box_information import BlackBoxInformation
import torch.nn.functional as F
from einops import rearrange

from pep_compass.models.battleamp.BattleAMPPredictor import PredictorBattleAMP

class BattleAMPBlackBox(AbstractBlackBox):
    def __init__(
        self,
        *,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        force_isolation: bool = False,
        device: str = "cpu",
    ):
        super().__init__(
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
            force_isolation=force_isolation,
        )
        
        self.device = device
        self.battleamp_predictor = PredictorBattleAMP(device=device)
        
        # BattleAMP returns a single prediction value, so no aggregation needed
        self.peptide_scorer = lambda x: -np.log2(self.battleamp_predictor.predict(x).flatten())

        self.maximize = False

        self.cache = []

    def get_black_box_info(self) -> BlackBoxInformation:
        return BlackBoxInformation(
            name="BattleAMP",
            max_sequence_length=25,
            aligned=False,
            fixed_length=False,
            deterministic=True,
            alphabet=list("ACDEFGHIKLMNPQRSTVWY"),
            log_transform_recommended=False,
            discrete=True,
            padding_token=" ",
        )

    def _black_box(self, x: np.ndarray, context: dict = None) -> np.ndarray:
        sequences = ["".join(seq) for seq in x]
        predictions = self.peptide_scorer(sequences)

        for i, seq in enumerate(sequences):
            self.cache.append((seq, predictions[i].item()))

        return predictions.reshape(-1, 1)
    
    def clear_cache(self):
        self.cache = []