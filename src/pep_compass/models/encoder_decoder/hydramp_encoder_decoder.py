import os
import torch
from pep_compass.models.hydramp.hydramp import HydrAMPDecoder, HydrAMPEncoder
from pep_compass.models.encoder_decoder.encoder_decoder import EncoderDecoder
from pep_compass.utils.sequence_utils import to_one_hot, translate_generated_peptide

class HydrAMPEncoderDecoder(EncoderDecoder):
    def __init__(
        self,
        jacobian_mode="strict",
        device="cpu",
        default_condition=torch.tensor([[1.0, 1.0]]),
        temp=1.0,
        jacobian_eps=0.1,
        field_eps=0.1,
    ):
        super().__init__(
            jacobian_mode=jacobian_mode, jacobian_eps=jacobian_eps, field_eps=field_eps
        )

        self.device = device

        self.default_condition = default_condition.to(device)
        self.temp = temp

        self.encoder = HydrAMPEncoder(device=device)
        self.decoder = HydrAMPDecoder(device=device)
        
        file_dir = os.path.dirname(os.path.abspath(__file__))
        weights_dir = f"{file_dir}/../hydramp/weights"
               
        self.encoder_load_state_dict(
            torch.load(f"{weights_dir}/encoder_weights.pickle", weights_only=True)
        )
        self.decoder_load_state_dict(
            torch.load(f"{weights_dir}/decoder_weights.pickle", weights_only=True)
        )
        

    def get_latent_dim(self):
        return 64

    def get_ambient_dim(self):
        return 525

    def encoder_load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict)

    def decoder_load_state_dict(self, state_dict):
        self.decoder.load_state_dict(state_dict)

    def decoder_forward(self, x, softmax_and_flatten=True):
        # Add batch dim if it is not present
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        decoder_input = x

        # Add condition to the input
        if x.shape[1] == self.get_latent_dim():
            if x.shape[0] > 1:
                condition = self.default_condition.repeat(x.shape[0], 1).to(self.device)
            else:
                condition = self.default_condition.to(self.device)
            decoder_input = torch.cat([x, condition], dim=-1)

        decoder_output = self.decoder(decoder_input)

        if softmax_and_flatten:
            return torch.softmax(decoder_output / self.temp, dim=-1).view(-1, 525)
        else:
            return decoder_output

    def decoder_forward_softmax(self, x):
        # Add batch dim if it is not present
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        decoder_input = x

        # Add condition to the input
        if x.shape[1] == self.get_latent_dim():
            if x.shape[0] > 1:
                condition = self.default_condition.repeat(x.shape[0], 1)
            else:
                condition = self.default_condition
            decoder_input = torch.cat([x, condition], dim=-1)

        decoder_output = self.decoder(decoder_input)

        return torch.softmax(decoder_output / self.temp, dim=-1)

    def decode_peptides(self, batch: torch.Tensor, batch_size=1):
        if batch.ndim == 1:
            batch = batch.unsqueeze(0)

        decoded_peptides = []
        for i in range(0, batch.shape[0], batch_size):
            z = batch[i : i + batch_size]
            decoded_logits = self.decoder_forward(z, softmax_and_flatten=False)
            decoded_peptides.extend(
                [
                    translate_generated_peptide(logits.unsqueeze(0))
                    for logits in decoded_logits
                ]
            )

        return decoded_peptides

    def decoder_log_softmax(self, x):

        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        if x.shape[1] == self.get_latent_dim():
            if x.shape[0] > 1:
                condition = self.default_condition.repeat(x.shape[0], 1)
            else:
                condition = self.default_condition
            decoder_input = torch.cat([x, condition], dim=-1)

        f = torch.nn.LogSoftmax(dim=-1)

        return f(self.decoder(decoder_input)).view(-1, 525)

    def encoder_forward(self, x):
        mean, std = self.encoder(x)
        return mean

    def encode_peptides_with_std(
        self, peptides: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        one_hot_peptides = [to_one_hot(peptide) for peptide in peptides]
        one_hot_tensor = torch.tensor(one_hot_peptides, device=self.device)
        mean, std = self.encoder(one_hot_tensor)
        return mean, std

    def encode_peptides(self, peptides: list[str]) -> torch.Tensor:
        one_hot_peptides = [to_one_hot(peptide) for peptide in peptides]
        return self.encoder_forward(torch.tensor(one_hot_peptides, device=self.device))