import torch
import torch.nn as nn


class VariationalAutoencoderNetwork(nn.Module):

    def __init__(self, inputNeurons=4, hiddenNeurons=3, outputNeurons=4):
        super(VariationalAutoencoderNetwork, self).__init__()

        self.encoderMean = nn.Linear(inputNeurons, hiddenNeurons)
        self.encoderStDev = nn.Linear(inputNeurons, hiddenNeurons)
        self.decoder = nn.Linear(hiddenNeurons, outputNeurons)

        self.activation = torch.relu

    def encode(self, x):

        return self.encoderMean(x), self.encoderStDev(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    #TODO: use reparametrization in the forward step, check the sampling operation

    # def forward(self, vinput):
    #     out = self.activation(self.encoder(vinput))
    #     out = (self.decoder(out))
    #
    #     return out
