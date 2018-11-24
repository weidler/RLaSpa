import torch
import torch.nn as nn


class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Linear(4, 3)
        self.decoder = nn.Linear(3, 4)

        self.activation = torch.sigmoid

    def forward(self, vinput):
        out = self.activation(self.encoder(vinput))
        out = (self.decoder(out))

        return out
