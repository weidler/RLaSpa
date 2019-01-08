import torch
import torch.nn as nn


class VariationalAutoencoderNetwork(nn.Module):

    def __init__(self, inputNeurons=4, hiddenNeurons=3, outputNeurons=4):
        super(VariationalAutoencoderNetwork, self).__init__()

        self.encoderMean = nn.Linear(inputNeurons, hiddenNeurons)
        self.encoderStDev = nn.Linear(inputNeurons, hiddenNeurons)
        self.decoder = nn.Linear(hiddenNeurons, outputNeurons)

        self.activation = torch.relu

    # Ok it took some time but I think I got the math behind.
    # for numerical stability we use the log of the variance (sigma squared)
    # then by taking the exponential of the log of the variance we got the standard deviation
    # e^(log(sigma^2)/2 = sigma, the standard deviation

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, input):
        mu = self.activation(self.encoderMean(input))
        logvar = self.activation(self.encoderStDev(input))
        z = self.reparameterize(mu, logvar)
        return torch.sigmoid(self.decoder(z)), mu, logvar
