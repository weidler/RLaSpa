import torch
import torch.nn as nn


class VariationalAutoencoderNetwork(nn.Module):

    def __init__(self, inputNeurons=4, midNeurons=3, hiddenNeurons=2, outputNeurons=4):
        super(VariationalAutoencoderNetwork, self).__init__()

        self.fullyConnected = nn.Linear(inputNeurons, midNeurons)
        self.encoderMean = nn.Linear(midNeurons, hiddenNeurons)
        self.encoderStDev = nn.Linear(midNeurons, hiddenNeurons)
        self.decodeFc = nn.Linear(hiddenNeurons, midNeurons)
        self.decoderOut = nn.Linear(midNeurons, outputNeurons)

        self.activation = torch.relu

    # Ok it took some time but I think I got the math behind.
    # for numerical stability we use the log of the variance (sigma squared)
    # then by taking the exponential of the log of the variance we got the standard deviation
    # e^(log(sigma^2)/2 = sigma, the standard deviation

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, input: torch.Tensor):
        # z1 = self.activation(self.fullyConnected(input.reshape(-1)))
        z1 = self.activation(self.fullyConnected(input))
        mu = self.encoderMean(z1)
        logvar = self.encoderStDev(z1)
        z2 = self.reparameterize(mu, logvar)
        out = self.activation(self.decodeFc(z2))
        return torch.sigmoid(self.decoderOut(out)), mu, logvar
