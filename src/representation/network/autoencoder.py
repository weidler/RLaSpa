import torch
import torch.nn as nn


class AutoencoderNetwork(nn.Module):

    def __init__(self, input_neurons=4, hidden_neurons=3, output_neurons=4):
        super(AutoencoderNetwork, self).__init__()

        self.encoder = nn.Linear(input_neurons, hidden_neurons)
        self.decoder = nn.Linear(hidden_neurons, output_neurons)

        self.activation = torch.sigmoid

    def forward(self, state: torch.Tensor):
        if state.dim() <= 1:
            raise ValueError(
                "Networks expect any input to be given as a batch. For single input, provide a batch of size 1.")

        out = self.activation(self.encoder(state))
        return self.decoder(out)
