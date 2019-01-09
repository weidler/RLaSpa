import torch
import torch.nn as nn
import torch.optim as optim
import random


class CerberusNetwork(nn.Module):

    def __init__(self, d_in, d_hidden, d_actions):
        super(CerberusNetwork, self).__init__()

        self.encoder = nn.Linear(d_in, d_hidden)

        self.decoder_reconstruction = nn.Linear(d_hidden, d_in)
        self.decoder_next = nn.Linear(d_hidden + d_actions, d_in)
        self.decoder_difference = nn.Linear(d_hidden + d_actions, d_in)

        self.activation = torch.tanh

    def forward(self, state, action):
        if state.dim() <= 1 or action.dim() <= 1:
            raise ValueError(
                "Networks expect any input to be given as a batch. For single input, provide a batch of size 1.")

        # encode current state and create latent space
        representation = self.activation(self.encoder(state))
        representation_plus_action = torch.cat((representation, action), 1)

        # decode heads
        reconstruction = self.activation(self.decoder_reconstruction(representation))
        next_state_prediction = self.activation(self.decoder_next(representation_plus_action))
        difference_prediction = self.activation(self.decoder_difference(representation_plus_action))

        return reconstruction, next_state_prediction, difference_prediction

if __name__ == "__main__":
    pass


