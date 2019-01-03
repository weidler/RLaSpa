import torch
import torch.nn as nn


class AutoencoderNetwork(nn.Module):

    def __init__(self, inputNeurons=4, hiddenNeurons=3, outputNeurons=4):
        super(AutoencoderNetwork, self).__init__()

        self.encoder = nn.Linear(inputNeurons, hiddenNeurons)
        self.decoder = nn.Linear(hiddenNeurons, outputNeurons)

        self.activation = torch.sigmoid

    def forward(self, vinput):
        out = self.activation(self.encoder(vinput))
        out = (self.decoder(out))

        return out
