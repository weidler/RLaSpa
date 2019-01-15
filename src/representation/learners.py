import torch
from torch import Tensor
from torch import nn, optim

from src.gym_custom_tasks.envs import ObstaclePathing
from src.representation.network.autoencoder import AutoencoderNetwork
from src.representation.network.cerberus import CerberusNetwork
from src.representation.network.janus import JanusAutoencoder
from src.representation.network.variational_autoencoder import VariationalAutoencoderNetwork
from src.representation.representation import _RepresentationLearner
from src.representation.visual.pixelencoder import JanusPixelEncoder, CerberusPixelEncoder, VariationalPixelEncoder, \
    CVAE, ConvolutionalNetwork

import matplotlib.pyplot as plt
import gym

class PassThrough(_RepresentationLearner):
    def __init__(self):
        pass

    def encode(self, state: Tensor) -> Tensor:
        return state

    def learn(self, state: Tensor, action: Tensor, reward: Tensor, next_state: Tensor) -> float:
        return 0.0


class Flatten(_RepresentationLearner):

    def __init__(self):
        pass

    def encode(self, state: Tensor) -> Tensor:
        return state.view(-1)

    def learn(self, state: Tensor, action: Tensor, reward: Tensor, next_state: Tensor) -> float:
        return 0.0


class SimpleAutoencoder(_RepresentationLearner):

    def __init__(self, d_states: int, d_actions: int, d_latent: int, lr: float = 0.1):
        # PARAMETERS
        self.d_states = d_states
        self.d_actions = d_actions
        self.d_latent = d_latent

        self.learning_rate = lr

        # NETWORK
        self.network = AutoencoderNetwork(d_states, d_latent, d_states)

        # PARTS
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.learning_rate)

    def encode(self, state: Tensor) -> Tensor:
        return self.network.activation(self.network.encoder(state))

    def learn(self, state: Tensor, action: Tensor, reward: Tensor, next_state: Tensor) -> float:
        # convert to tensor if necessary

        self.optimizer.zero_grad()
        out = self.network(state)
        loss = self.criterion(out, state)
        loss.backward()

        self.optimizer.step()
        return loss.data.item()


class VariationalAutoencoder(_RepresentationLearner):

    def __init__(self, d_states: int, d_actions: int, d_middle: int, d_latent: int, lr: float = 0.005): # 1e-3 is the one originally used
        # PARAMETERS
        self.d_states = d_states
        self.d_actions = d_actions
        self.d_latent = d_latent

        self.learning_rate = lr

        # NETWORK
        self.network = VariationalAutoencoderNetwork(d_states, d_middle, d_latent, d_states)

        # PARTS
        # self.criterion = nn.MSELoss()
        # self.criterion = nn.functional.binary_cross_entropy()
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        # self.optimizer = optim.SGD(self.network.parameters(), lr=self.learning_rate)

    def loss_function(self, recon_x, x_tens, mu, logvar) -> float:
        BCE = nn.functional.binary_cross_entropy(recon_x, x_tens.view(-1, self.d_states), reduction='sum')
        # BCE = self.criterion(recon_x, x_tens.view(-1, self.d_states), reduction='sum')
        # MSE = self.criterion(recon_x, x_tens)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD
        # return MSE + KLD

    def encode(self, state: Tensor) -> Tensor:
        # z1 = self.network.activation(self.network.fullyConnected(state.reshape(-1)))
        z1 = self.network.activation(self.network.fullyConnected(state))
        mu = self.network.encoderMean(z1)
        logvar = self.network.encoderStDev(z1)
        z2 = self.network.reparameterize(mu, logvar)
        return z2

    def learn(self, state: Tensor, action: Tensor, reward: Tensor, next_state: Tensor) -> float:
        self.optimizer.zero_grad()
        out, mu, logvar = self.network(state)
        loss = self.loss_function(out, state, mu, logvar)
        loss.backward()

        self.optimizer.step()
        return loss.data.item()


class VariationalAutoencoderPixel(_RepresentationLearner):

    def __init__(self, width: int, height: int, n_middle: int, n_hidden: int, lr: float = 1e-3): # 1e-3 is the one originally used
        # PARAMETERS
        self.width = width
        self.height = height
        self.n_middle = n_middle
        self.n_hidden = n_hidden

        self.learning_rate = lr

        # NETWORK
        self.network = VariationalPixelEncoder(
            width=self.width,
            height=self.height,
            n_middle=self.n_middle,
            n_hidden=self.n_hidden
        )

        # PARTS
        # self.criterion = nn.MSELoss()
        # self.criterion = nn.functional.binary_cross_entropy()
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        # self.optimizer = optim.SGD(self.network.parameters(), lr=self.learning_rate)

    def loss_function(self, recon_x, x_tens, mu, logvar) -> float:
        BCE = nn.functional.binary_cross_entropy(recon_x, x_tens.view(-1, self.width * self.height), reduction='sum')
        # BCE = self.criterion(recon_x, x_tens.view(-1, self.d_states), reduction='sum')
        # MSE = self.criterion(recon_x, x_tens)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD
        # return MSE + KLD

    def visualize_output(self, state: Tensor, action: Tensor, next_state: Tensor):
        reconstruction, mu, logvar = self.network(torch.unsqueeze(state, 0))
        plt.clf()
        plt.imshow(torch.squeeze(state).tolist() + torch.squeeze(reconstruction).tolist(), cmap="binary",
                   origin="upper")
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.draw()
        plt.pause(0.001)

    def encode(self, state: Tensor) -> Tensor:
        z1 = self.network.activation(self.network.fullyConnected(state.reshape(-1)))
        # z1 = self.network.activation(self.network.fullyConnected(state))
        mu = self.network.encoderMean(z1)
        logvar = self.network.encoderStDev(z1)
        z2 = self.network.reparameterize(mu, logvar)
        return z2

    def learn(self, state: Tensor, action: Tensor, reward: Tensor, next_state: Tensor) -> float:
        self.optimizer.zero_grad()
        out, mu, logvar = self.network(state)
        loss = self.loss_function(out, state, mu, logvar)
        loss.backward()

        self.optimizer.step()
        return loss.data.item()


class ConvolutionalPixel(_RepresentationLearner):

    def __init__(self, n_output: int, lr: float = 1e-3):

        self.n_output = n_output
        self.learning_rate = lr

        self.network = ConvolutionalNetwork(self.n_output)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.RMSprop(self.network.parameters(), self.learning_rate)

    def encode(self, state: Tensor) -> Tensor:
        return self.network.forward(state)

    def learn(self, state: Tensor, action: Tensor, reward: Tensor, next_state: Tensor) -> float:
        # convert to tensor if necessary

        self.optimizer.zero_grad()
        out = self.network(state)
        loss = self.criterion(out, state)
        loss.backward()

        self.optimizer.step()
        return loss.data.item()


class CVAEPixel(_RepresentationLearner):

    def __init__(self, width: int, height: int, n_middle: int, n_hidden: int, lr: float = 1e-3): # 1e-3 is the one originally used
        # PARAMETERS
        self.width = width
        self.height = height
        self.n_middle = n_middle
        self.n_hidden = n_hidden

        self.learning_rate = lr

        # NETWORK
        self.network = CVAE(
            width=self.width,
            height=self.height,
            n_middle=self.n_middle,
            n_hidden=self.n_hidden
        )

        # PARTS
        # self.criterion = nn.MSELoss()
        # self.criterion = nn.functional.binary_cross_entropy()
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        # self.optimizer = optim.SGD(self.network.parameters(), lr=self.learning_rate)

    def loss_function(self, recon_x, x_tens, mu, logvar) -> float:
        BCE = nn.functional.binary_cross_entropy(recon_x, x_tens.view(-1, self.width * self.height), reduction='sum')
        # BCE = self.criterion(recon_x, x_tens.view(-1, self.d_states), reduction='sum')
        # MSE = self.criterion(recon_x, x_tens)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD
        # return MSE + KLD

    def visualize_output(self, state: Tensor):
        reconstruction, mu, logvar = self.network(torch.unsqueeze(state, 0))
        plt.clf()
        plt.imshow(torch.squeeze(state).tolist() + torch.squeeze(reconstruction).tolist(), cmap="binary",
                   origin="upper")
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.draw()
        plt.pause(0.001)

    def encode(self, x: Tensor):
        conv1 = self.network.relu(self.network.bn1(self.network.conv1(x.reshape(16, 1, 3, 3))))
        conv2 = self.network.relu(self.network.bn2(self.network.conv2(conv1)))
        conv3 = self.network.relu(self.network.bn3(self.network.conv3(conv2)))
        conv4 = self.network.relu(self.network.bn4(self.network.conv4(conv3))).view(-1, )

        fc1 = self.network.relu(self.network.fc_bn1(self.network.fc1(conv4)))
        return self.network.fc21(fc1), self.network.fc22(fc1)

    def learn(self, state: Tensor, action: Tensor, reward: Tensor, next_state: Tensor) -> float:
        self.optimizer.zero_grad()
        out, mu, logvar = self.network(state)
        loss = self.loss_function(out, state, mu, logvar)
        loss.backward()

        self.optimizer.step()
        return loss.data.item()


class Janus(_RepresentationLearner):

    def __init__(self, d_states: int, d_actions: int, d_latent: int, lr: float = 0.1):
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

        # PARTS
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.learning_rate)

    def encode(self, state: Tensor) -> Tensor:
        return self.network.activation(self.network.encoder(state))

    def learn(self, state: Tensor, action: Tensor, reward: Tensor, next_state: Tensor) -> float:
        # convert to tensor if necessary
        target_tensor = torch.cat((state, next_state), 0)

        self.optimizer.zero_grad()
        out = self.network(state, action)
        loss = self.criterion(out, target_tensor)
        loss.backward()

        self.optimizer.step()
        return loss.data.item()


class JanusPixel(_RepresentationLearner):
    def __init__(self, width: int, height: int, n_actions: int, n_hidden: int, lr: float = 0.1):
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

        # PARTS
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.learning_rate)

    # def visualize_output(self, state: Tensor, action: Tensor, next_state: Tensor):
    #     reconstruction, next_state_construction = self.network(torch.unsqueeze(state, 0),
    #                                                            torch.unsqueeze(action, 0))
    #     plt.imshow(torch.squeeze(state).tolist() + torch.squeeze(reconstruction).tolist(), cmap="binary",
    #                origin="upper")
    #     plt.gca().axes.get_xaxis().set_visible(False)
    #     plt.gca().axes.get_yaxis().set_visible(False)
    #     plt.show()

    def encode(self, state: Tensor) -> Tensor:
        return self.network.activation(self.network.encoder(state.view(-1)))

    def learn(self, state: Tensor, action: Tensor, reward: Tensor, next_state: Tensor) -> float:
        self.optimizer.zero_grad()
        reconstruction, next_state_construction = self.network(state, action)

        # Loss
        reconstruction_loss = self.criterion(reconstruction, state)
        next_state_loss = self.criterion(next_state_construction, next_state)
        total_loss = sum([reconstruction_loss, next_state_loss])
        total_loss.backward()

        self.optimizer.step()
        return total_loss.data.item()


class Cerberus(_RepresentationLearner):

    def __init__(self, d_states: int, d_actions: int, d_latent: int, lr: float = 0.1):
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

        # PARTS
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.learning_rate)

    def encode(self, state: Tensor) -> Tensor:
        return self.network.activation(self.network.encoder(state))

    def learn(self, state: Tensor, action: Tensor, reward: Tensor, next_state: Tensor) -> float:
        difference = state != next_state

        self.optimizer.zero_grad()
        reconstruction, next_state_prediction, difference_prediction = self.network(state, action)

        # Loss
        reconstruction_loss = self.criterion(reconstruction, state)
        next_state_prediction_loss = self.criterion(next_state_prediction, next_state)
        difference_prediction_loss = self.criterion(difference_prediction, difference)
        total_loss = sum([reconstruction_loss, next_state_prediction_loss, difference_prediction_loss])

        total_loss.backward()

        self.optimizer.step()
        return total_loss.data.item()


class CerberusPixel(_RepresentationLearner):
    def __init__(self, width: int, height: int, n_actions: int, n_hidden: int, lr: float = 0.1):
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

        # PARTS
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.learning_rate)

    def encode(self, state: Tensor) -> Tensor:
        return self.network.activation(self.network.encoder(state.view(-1)))

    def learn(self, state: Tensor, action: Tensor, reward: Tensor, next_state: Tensor) -> float:
        difference_target = state != next_state

        self.optimizer.zero_grad()
        reconstruction, next_state_construction, difference = self.network(state, action)

        # Loss
        reconstruction_loss = self.criterion(reconstruction, state)
        next_state_loss = self.criterion(next_state_construction, next_state)
        difference_loss = self.criterion(difference, difference_target.float())
        total_loss = sum([reconstruction_loss, next_state_loss, difference_loss])
        total_loss.backward()

        self.optimizer.step()
        return total_loss.data.item()

    def visualize_output(self, state: Tensor, action: Tensor, next_state: Tensor):
        difference_tensor = (state != next_state)
        reconstruction, next_state_reconstruction, difference = self.network(torch.unsqueeze(state, 0),
                                                                           torch.unsqueeze(action, 0))
        plt.clf()

        vertical_seperator = [[1 for _ in range(len(torch.squeeze(state).tolist()[1]))]]
        reconstruction_image = torch.squeeze(state).tolist() + vertical_seperator + torch.squeeze(reconstruction).tolist()
        next_state_reconstruction_image = torch.squeeze(next_state).tolist() + vertical_seperator + torch.squeeze(next_state_reconstruction).tolist()
        difference_image = torch.squeeze(difference_tensor).tolist() + vertical_seperator + torch.squeeze(difference).tolist()
        horizontal_seperator = [[1] for _ in range(len(reconstruction_image))]
        full_image = [l[0] + l[1] + l[2] + l[3] + l[4] for l in list(zip(reconstruction_image, horizontal_seperator, next_state_reconstruction_image, horizontal_seperator, difference_image))]

        plt.imshow(full_image, cmap="binary", origin="upper")

        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.draw()
        plt.pause(0.001)

if __name__ == "__main__":
    # ae = Cerberus(d_states=5, d_actions=2, d_latent=5)
    ae = VariationalAutoencoder(d_states=900, d_actions=2, d_middle=400, d_latent=20)
    # ae = VariationalAutoencoder(d_states=8, d_actions=2, d_middle=4, d_latent=2)

    # for i in range(2500):
    #     # sample = [1, 2, 3, 4, 5]
    #
    #     # '''One hot encoded vector reconstruction test'''
    #     # samples = numpy.eye(8, dtype=int)
    #     # random.shuffle(samples)
    #     # sample = samples[random.randint(0, 7)]
    #
    #     '''Obstacle pathing test'''
    #
    #     env = ObstaclePathing(30, 30,
    #                           [[0, 18, 18, 21],
    #                            [21, 24, 10, 30]],
    #                           True
    #                           )
    #     sample = env.static_pixels
    #     loss = ae.learn(torch.Tensor(sample), None, None, None)
    #     if i % 100 == 0:
    #         print("Epoch ", i, " loss: ", loss)
    # #
    # for i in range(1):
    #     # sample = [1, 2, 3, 4, 5]
    #
    #     # '''One hot encoded vector reconstruction test'''
    #     # random.shuffle(sample)
    #     # samples = numpy.eye(8, dtype=int)
    #     # random.shuffle(samples)
    #     # sample = samples[random.randint(0, 7)]
    #     # # print(f"{sample} --> {[round(e) for e in ae.network(torch.Tensor(sample).float(), torch.Tensor([1,2]).float())[0].tolist()]}")
    #     # print(f"{sample} --> {[e for e in ae.network(torch.Tensor(sample))]}")
    #
    # print()
