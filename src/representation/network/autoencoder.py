import torch
import torch.nn as nn


class AutoencoderNetwork(nn.Module):

    def __init__(self, input_neurons=4, hidden_neurons=3, output_neurons=4):
        super(AutoencoderNetwork, self).__init__()

        self.encoder = nn.Linear(input_neurons, hidden_neurons)
        self.decoder = nn.Linear(hidden_neurons, output_neurons)

        self.activation = torch.sigmoid

    def forward(self, vinput):
        out = self.activation(self.encoder(vinput))
        return self.decoder(out)
