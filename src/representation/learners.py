import random

import numpy
import torch
from numpy.core.multiarray import ndarray
from torch import nn, optim

from src.representation.network.janus import JanusAutoencoder
from src.representation.network.autoencoder import AutoencoderNetwork
from src.representation.network.cerberus import CerberusNetwork
from src.representation.network.variational_autoencoder import VariationalAutoencoderNetwork
from src.representation.visual.pixelencoder import JanusPixelEncoder, CerberusPixelEncoder
from src.representation.representation import _RepresentationLearner


def cast_float_tensor(o: object):
    # if state is given as list, convert to required tensor
    if isinstance(o, list):
        o = torch.Tensor(o).float()
    # if state is given as ndarray, convert to required tensor
    elif isinstance(o, ndarray):
        o = torch.from_numpy(o).float()
    # if object is an int
    elif isinstance(o, int):
        o = torch.Tensor([o]).float()

    # check unknown cases
    if not isinstance(o, torch.Tensor):
        raise ValueError(f"Cannot cast Tensor on type {type(o)}")

    return o


class PassThrough(_RepresentationLearner):
     def __init__(self):
         pass

     def encode(self, state):
         return state

     def learn(self, state, action, reward, next_state, remember=True):
         return 0


class Flatten(_RepresentationLearner):

    def __init__(self):
        pass

    def encode(self, state):
        state_tensor = cast_float_tensor(state)
        return state_tensor.view(-1)

    def learn(self, state, action, reward, next_state, remember=True):
        return 0


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


class VariationalAutoencoder(_RepresentationLearner):

    def __init__(self, d_states, d_actions, d_latent, lr=0.05): # 1e-3 is the one originally used
        # PARAMETERS
        self.d_states = d_states
        self.d_actions = d_actions
        self.d_latent = d_latent

        self.learning_rate = lr

        # NETWORK
        self.network = VariationalAutoencoderNetwork(d_states, d_latent, d_states)

        # TRAINING SAMPLES
        self.backup_history = []

        # PARTS
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        # self.optimizer = optim.SGD(self.network.parameters(), lr=self.learning_rate)

    def loss_function(self, recon_x, x_tens, mu, logvar):
        # BCE = nn.functional.binary_cross_entropy(recon_x, x_tens.view(-1, self.d_states), reduction='sum')
        # MSE = nn.MSELoss(recon_x, x_tens)
        MSE = self.criterion(recon_x, x_tens)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # return BCE + KLD
        return MSE + KLD

    def encode(self, state):
        state = cast_float_tensor(state)
        return self.activation(self.encoderMean(state)), self.activation(self.encoderStDev(state))

    def learn(self, state, action=None, reward=None, next_state=None, remember=True):
        # remember sample in history
        if remember:
            self.backup_history.append((state, action, reward, next_state))

        # convert to tensor if necessary
        state_tensor = cast_float_tensor(state)

        self.optimizer.zero_grad()
        out, mu, logvar = self.network(state_tensor)
        loss = self.loss_function(out, state_tensor, mu, logvar)
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
        self.network = JanusAutoencoder(
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


class JanusPixel(_RepresentationLearner):
    def __init__(self, width, height, n_actions, n_hidden, lr=0.1):
        # PARAMETERS
        self.width = width
        self.height = height
        self.n_actions = n_actions
        self.n_hidden = n_hidden

        self.learning_rate = lr

        # NETWORK
        self.network = JanusPixelEncoder(
            width=self.width,
            height=self.height,
            n_actions=self.n_actions,
            n_hidden=self.n_hidden
        )
        self.one_hot_actions = numpy.eye(n_actions)

        # TRAINING SAMPLES
        self.backup_history = []

        # PARTS
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.learning_rate)

    def encode(self, state):
        state = cast_float_tensor(state.reshape(-1))
        return self.network.activation(self.network.encoder(state))

    def learn(self, state, action, reward, next_state, remember=True):
        action = self.one_hot_actions[action]
        # remember sample in history
        if remember:
            self.backup_history.append((state, action, reward, next_state))

        # convert to tensor if necessary
        state_tensor = cast_float_tensor(state)
        action_tensor = cast_float_tensor(action)
        next_state_tensor = cast_float_tensor(next_state)

        self.optimizer.zero_grad()
        reconstruction, next_state_construction = self.network(state_tensor, action_tensor)

        # Loss
        reconstruction_loss = self.criterion(reconstruction, state_tensor)
        next_state_loss = self.criterion(next_state_construction, next_state_tensor)
        total_loss = sum([reconstruction_loss, next_state_loss])
        total_loss.backward()

        self.optimizer.step()
        return total_loss.data.item()


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

    def learn(self, state, action=None, reward=None, next_state=None, remember=True):
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


class CerberusPixel(_RepresentationLearner):
    def __init__(self, width, height, n_actions, n_hidden, lr=0.1):
        # PARAMETERS
        self.width = width
        self.height = height
        self.n_actions = n_actions
        self.n_hidden = n_hidden

        self.learning_rate = lr

        # NETWORK
        self.network = CerberusPixelEncoder(
            width=self.width,
            height=self.height,
            n_actions=self.n_actions,
            n_hidden=self.n_hidden
        )
        self.one_hot_actions = numpy.eye(n_actions)

        # TRAINING SAMPLES
        self.backup_history = []

        # PARTS
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.learning_rate)

    def encode(self, state):
        state = cast_float_tensor(state.reshape(-1))
        return self.network.activation(self.network.encoder(state))

    def learn(self, state, action, reward, next_state, remember=True):
        action = self.one_hot_actions[action]
        # remember sample in history
        if remember:
            self.backup_history.append((state, action, reward, next_state))

        # convert to tensor if necessary
        state_tensor = cast_float_tensor(state)
        action_tensor = cast_float_tensor(action)
        next_state_tensor = cast_float_tensor(next_state)
        difference_tensor = cast_float_tensor((numpy.array(state) != numpy.array(next_state)).astype(int))

        self.optimizer.zero_grad()
        reconstruction, next_state_construction, difference = self.network(state_tensor, action_tensor)

        # Loss
        reconstruction_loss = self.criterion(reconstruction, state_tensor)
        next_state_loss = self.criterion(next_state_construction, next_state_tensor)
        difference_loss = self.criterion(difference, difference_tensor)
        total_loss = sum([reconstruction_loss, next_state_loss, difference_loss])
        total_loss.backward()

        self.optimizer.step()
        return total_loss.data.item()


if __name__ == "__main__":
    # ae = Cerberus(d_states=5, d_actions=2, d_latent=5)
    ae = VariationalAutoencoder(d_states=5, d_actions=2, d_latent=5)
    for i in range(100000):
        sample = [1, 2, 3, 4, 5]
        random.shuffle(sample)
        loss = ae.learn(sample, [1, 2], 1, sample)
        if i % 1000 == 0: print(loss)

    for i in range(10):
        sample = [1, 2, 3, 4, 5]
        random.shuffle(sample)
        # print(f"{sample} --> {[round(e) for e in ae.network(torch.Tensor(sample).float(), torch.Tensor([1,2]).float())[0].tolist()]}")
        print(f"{sample} --> {[round(numpy.array(e.tolist())) for e in ae.network(torch.Tensor(sample).float())]}")

    print()
