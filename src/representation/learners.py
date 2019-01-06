import random

import numpy
import torch
from numpy.core.multiarray import ndarray
from torch import nn, optim

from src.representation.network.autoencoder import AutoencoderNetwork
from src.representation.network.cerberus import CerberusNetwork
from src.representation.representation import _RepresentationLearner
from src.representation.siamese_autoencoder import SiameseAutoencoder


def cast_float_tensor(o: object):
    # if state is given as list, convert to required tensor
    if isinstance(o, list):
        o = torch.Tensor(o).float()
    # if state is given as ndarray, convert to required tensor
    elif isinstance(o, ndarray):
        o = torch.from_numpy(o).float()

    # check unknown cases
    if not isinstance(o, torch.Tensor):
        raise ValueError(f"Cannot cast Tensor on type {type(o)}")

    return o


class SimpleAutoencoder(_RepresentationLearner):

    def __init__(self, d_states, d_actions, d_latent, lr=0.05):
        # PARAMETERS
        self.d_states = d_states
        self.d_actions = d_actions
        self.d_latent = d_latent

        self.learning_rate = lr

        # NETWORK
        self.network = AutoencoderNetwork(d_states, d_latent, d_states)

        # TRAINING SAMPLES
        self.backup_history = []

        # PARTS
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.learning_rate)

    def encode(self, state):
        state = cast_float_tensor(state)
        return self.network.activation(self.network.encoder(state))

    def learn(self, state, action=None, reward=None, next_state=None, remember=True):
        # remember sample in history
        if remember:
            self.backup_history.append((state, action, reward, next_state))

        # convert to tensor if necessary
        state_tensor = cast_float_tensor(state)

        self.optimizer.zero_grad()
        out = self.network(state_tensor)
        loss = self.criterion(out, state_tensor)  # TODO not sure if it is ok to use same tensor or if we need to copy
        loss.backward()

        self.optimizer.step()
        return loss.data.item()


class Janus(_RepresentationLearner):

    def __init__(self, d_states, d_actions, d_latent, lr=0.1):
        # PARAMETERS
        self.d_states = d_states
        self.d_actions = d_actions
        self.d_latent = d_latent

        self.learning_rate = lr

        # NETWORK
        self.network = SiameseAutoencoder(
            inputNeurons=d_states,
            hiddenNeurons=d_latent,
            outputNeurons=d_states,
            actionDim=d_actions
        )

        # TRAINING SAMPLES
        self.backup_history = []

        # PARTS
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.learning_rate)

    def encode(self, state):
        state = cast_float_tensor(state)
        return self.network.activation(self.network.encoder(state))

    def learn(self, state, action, reward, next_state, remember=True):
        # remember sample in history
        if remember:
            self.backup_history.append((state, action, reward, next_state))

        # convert to tensor if necessary
        state_tensor = cast_float_tensor(state)
        action_tensor = cast_float_tensor(action)
        next_state_tensor = cast_float_tensor(next_state)
        target_tensor = torch.cat((state_tensor, next_state_tensor), 0)

        self.optimizer.zero_grad()
        out = self.network(state_tensor, action_tensor)
        loss = self.criterion(out, target_tensor)
        loss.backward()

        self.optimizer.step()
        return loss.data.item()


class Cerberus(_RepresentationLearner):

    def __init__(self, d_states, d_actions, d_latent, lr=0.1):
        # PARAMETERS
        self.d_states = d_states
        self.d_actions = d_actions
        self.d_latent = d_latent

        self.learning_rate = lr

        # NETWORK
        self.network = CerberusNetwork(
            d_in=d_states,
            d_hidden=d_latent,
            d_actions=d_actions
        )

        # TRAINING SAMPLES
        self.backup_history = []

        # PARTS
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.learning_rate)

    def encode(self, state):
        state = cast_float_tensor(state)
        return self.network.activation(self.network.encoder(state))

    def learn(self, state, action, reward, next_state, remember=True):
        # remember sample in history
        if remember:
            self.backup_history.append((state, action, reward, next_state))

        # convert to tensor if necessary
        state_tensor = cast_float_tensor(state)
        action_tensor = cast_float_tensor(action)
        next_state_tensor = cast_float_tensor(next_state)
        difference_tensor = cast_float_tensor((numpy.array(state) != numpy.array(next_state)).astype(int))

        self.optimizer.zero_grad()
        reconstruction, next_state_prediction, difference_prediction = self.network(state_tensor, action_tensor)

        # Loss
        reconstruction_loss = self.criterion(reconstruction, state_tensor)
        next_state_prediction_loss = self.criterion(next_state_prediction, next_state_tensor)
        difference_prediction_loss = self.criterion(difference_prediction, difference_tensor)
        total_loss = sum([reconstruction_loss, next_state_prediction_loss, difference_prediction_loss])

        total_loss.backward()

        self.optimizer.step()
        return total_loss.data.item()


if __name__ == "__main__":
    ae = Cerberus(d_states=5, d_actions=2, d_latent=5)

    for i in range(100000):
        sample = [1, 2, 3, 4, 5]
        random.shuffle(sample)
        loss = ae.learn(sample, [1, 2], 1, sample)
        if i % 1000 == 0: print(loss)

    for i in range(10):
        sample = [1, 2, 3, 4, 5]
        random.shuffle(sample)
        print(f"{sample} --> {[round(e) for e in ae.network(torch.Tensor(sample).float(), torch.Tensor([1,2]).float())[0].tolist()]}")

    print()
