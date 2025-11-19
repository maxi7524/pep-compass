import numpy as np
import torch
from poli.core.black_box_information import BlackBoxInformation
from poli_baselines.core.abstract_solver import AbstractBlackBox

from pep_compass.models.encoder_decoder.hydramp_encoder_decoder import \
    HydrAMPEncoderDecoder


class HydrAMPBlackBoxWrapper(AbstractBlackBox):
    def __init__(
        self,
        *,
        black_box: AbstractBlackBox,
        device: str = "cpu",
        jacobian_eps: float,
        field_eps: float,
    ):
        super().__init__(
            batch_size=black_box.batch_size,
            parallelize=black_box.parallelize,
            num_workers=black_box.num_workers,
            evaluation_budget=black_box.evaluation_budget,
            force_isolation=black_box.force_isolation,
        )
        
        self.wrapped_black_box = black_box
        
        self.encoder_decoder = HydrAMPEncoderDecoder(
            device=device,
            default_condition=torch.tensor([1, 1], device=device),
            temp=1,
            jacobian_mode="approx",
            jacobian_eps=jacobian_eps,
            field_eps=field_eps,
        )

        self.cache = [] 

        self.shift = 0.0
        
        self.maximize = black_box.maximize
        
    def set_shift(self, shift: float):
        self.shift = shift

    def get_black_box_info(self) -> BlackBoxInformation:
        black_box_info = self.wrapped_black_box.get_black_box_info()
        
        black_box_info.name = "HydrAMPWrapped" + black_box_info.name
        black_box_info.discrete = False 
        
        return black_box_info

    def _black_box(self, x: np.ndarray, context: dict = None) -> np.ndarray:
        x_tensor = torch.tensor(x, device=self.encoder_decoder.device)
        decoded_peptides = self.encoder_decoder.decode_peptides(x_tensor.to(torch.float32))
        
        predictions = self.wrapped_black_box._black_box(decoded_peptides)
        
        if context is not None and isinstance(context, dict):
            context['sequences'] = decoded_peptides
        
        for i, seq in enumerate(decoded_peptides):
            self.cache.append((x_tensor[i].tolist(), seq, predictions[i].item()))

        return predictions.reshape(-1, 1) + self.shift  # Reshape to match the expected output shape

    def clear_cache(self):
        self.cache = []
