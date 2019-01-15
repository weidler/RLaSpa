import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable


class PixelEncoder(torch.nn.Module):

    def __init__(self):
        super(PixelEncoder, self).__init__()
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1, return_indices=True)

        self.encoder = torch.nn.Linear(225, 50)
        self.decoder = torch.nn.Linear(50, 225)

        self.unpool = torch.nn.MaxUnpool2d(kernel_size=2, stride=2, padding=1)
        self.unconv = torch.nn.ConvTranspose2d(1, 1, kernel_size=3, stride=1, padding=0)

    def forward(self, input_tensor: Tensor) -> Tensor:
        convoluted = self.conv(input_tensor)
        pooled, indices = self.pool(convoluted)
        original_shape = pooled.shape

        encoded = self.encoder(pooled.view(1, -1))
        decoded = self.decoder(encoded)
        deflattened = decoded.reshape(original_shape)

        unpooled = self.unpool(deflattened, indices)
        unconved = self.unconv(unpooled)

        return unconved


class SimplePixelEncoder(torch.nn.Module):

    def __init__(self, width: int, height: int, repr_size: int):
        super(SimplePixelEncoder, self).__init__()

        self.encoder = torch.nn.Linear(width * height, repr_size)
        self.decoder = torch.nn.Linear(repr_size, width * height)

        self.activation = torch.sigmoid

    def forward(self, input_tensor: Tensor) -> Tensor:
        original_shape = input_tensor.shape

        # Reshape IMAGE -> VECTOR
        flattened = input_tensor.view(1, -1)

        encoded = self.activation(self.encoder(flattened))
        decoded = self.decoder(encoded)

        # Reshape VECTOR -> IMAGE
        deflattened = decoded.reshape(original_shape)

        return deflattened


class VariationalPixelEncoder(torch.nn.Module):

    def __init__(self, width: int, height: int, n_middle: int, n_hidden: int=10):
        super(VariationalPixelEncoder, self).__init__()

        self.fullyConnected = nn.Linear(width * height, n_middle)
        self.encoderMean = nn.Linear(n_middle, n_hidden)
        self.encoderStDev = nn.Linear(n_middle, n_hidden)
        self.decodeFc = nn.Linear(n_hidden, n_middle)
        self.decoderOut = nn.Linear(n_middle, width * height)

        self.activation = torch.relu

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, state):
        original_shape = state.shape

        # Reshape IMAGE -> VECTOR
        flattened = state.view(state.shape[0], -1)

        z1 = self.activation(self.fullyConnected(flattened))
        mu = self.encoderMean(z1)
        logvar = self.encoderStDev(z1)
        z2 = self.reparameterize(mu, logvar)
        mid_out = self.activation(self.decodeFc(z2))
        out = torch.sigmoid(self.decoderOut(mid_out))
        # Reshape VECTOR -> IMAGE
        deflattened_out = out.reshape(original_shape)

        return deflattened_out, mu, logvar


class ConvolutionalNetwork(torch.nn.Module):

    def __init__(self, out_features: int = 512):
        super(ConvolutionalNetwork, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128*3*3, out_features)

        # Decoder
        self.fc2 = nn.Linear(out_features, 128*3*3)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0)
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=8, stride=4, padding=1)

        # Activation
        self.activation = nn.ReLU()

    def forward(self, input: Tensor):
        # DECODER
        input = input.view(-1, 1, 30, 30)  # the 1 is the channel
        conv1 = self.activation(self.conv1(input))
        conv2 = self.activation(self.conv2(conv1))
        conv3 = self.activation(self.conv3(conv2))
        flatten = conv3.view(input.size(0), -1)

        latent = self.fc1(flatten)

        unflatten = self.activation(self.fc2(latent)).view(-1, 128, 3, 3)
        deconv1 = self.activation(self.deconv1(unflatten))
        deconv2 = self.activation(self.deconv2(deconv1))
        deconv3 = self.activation(self.deconv3(deconv2))

        return deconv3.view(-1, 30, 30)


class CVAE(torch.nn.Module):

    def __init__(self, width: int, height: int, n_middle: int, n_hidden: int=10):
        super(CVAE, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(128 * 99 * 99, 512)
        # self.fc_bn1 = nn.BatchNorm1d(512)
        self.fc21 = nn.Linear(512, 512)
        # self.fc22 = nn.Linear(512, 512)

        # Decoder
        self.fc3 = nn.Linear(512, 512)
        self.fc_bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 8 * 8 * 16)
        self.fc_bn4 = nn.BatchNorm1d(8 * 8 * 16)

        self.conv5 = nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(32)
        self.conv7 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(16)
        self.conv8 = nn.ConvTranspose2d(16, 3 * 256, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU()

    def encode(self, x):
        conv1 = self.relu(self.bn1(self.conv1(x)))
        conv2 = self.relu(self.bn2(self.conv2(conv1)))
        conv3 = self.relu(self.bn3(self.conv3(conv2)))
        conv4 = self.relu(self.bn4(self.conv4(conv3))).view(-1, 20*30)

        fc1 = self.relu(self.fc_bn1(self.fc1(conv4)))
        return self.fc21(fc1), self.fc22(fc1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3))).view(-1, 16, 8, 8)

        conv5 = self.relu(self.bn5(self.conv5(fc4)))
        conv6 = self.relu(self.bn6(self.conv6(conv5)))
        conv7 = self.relu(self.bn7(self.conv7(conv6)))
        return self.conv8(conv7).view(-1, 256, 3, 32, 32)

    def forward(self, x: Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class JanusPixelEncoder(torch.nn.Module):

    def __init__(self, width: int, height: int, n_actions: int, n_hidden: int=10):
        super(JanusPixelEncoder, self).__init__()

        self.encoder = torch.nn.Linear(width * height, n_hidden)

        # decoderState decodes current state from state latent space
        self.decoderState = torch.nn.Linear(n_hidden, width * height)
        # decoderNextState decodes next state from state latent space + action
        self.decoderNextState = torch.nn.Linear(n_hidden + n_actions, width * height)

        self.activation = torch.sigmoid

    def forward(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):
        original_shape = state.shape

        # Reshape IMAGE -> VECTOR
        flattened = state.view(state.shape[0], -1)

        # encode current state and create latent space
        latent_space = self.activation(self.encoder(flattened))
        # decode current state
        outState = (self.decoderState(latent_space))
        # Reshape VECTOR -> IMAGE
        deflattened_reconstruction = outState.reshape(original_shape)

        # append action to latent space
        latent_space_action = torch.cat((latent_space, action), 1)
        # decode next state from latent space with action
        outNextState = self.decoderNextState(latent_space_action)
        # Reshape VECTOR -> IMAGE
        deflattened_next_state = outNextState.reshape(original_shape)

        return deflattened_reconstruction, deflattened_next_state


class CerberusPixelEncoder(torch.nn.Module):

    def __init__(self, width: int, height: int, n_actions: int, n_hidden: int=10):
        super(CerberusPixelEncoder, self).__init__()

        self.encoder = torch.nn.Linear(width * height, n_hidden)

        # decoderState decodes current state from state latent space
        self.decoderState = torch.nn.Linear(n_hidden, width * height)
        # decoderNextState decodes next state from state latent space + action
        self.decoderNextState = torch.nn.Linear(n_hidden + n_actions, width * height)
        # decoderDifference decodes the difference between the state and the next state
        self.decoderDifference = torch.nn.Linear(n_hidden + n_actions, width * height)

        self.activation = torch.sigmoid

        self.note = "HELLO"

    def forward(self, state: Tensor, action: Tensor) -> (Tensor, Tensor, Tensor):
        original_shape = state.shape

        # Reshape IMAGE -> VECTOR
        flattened = state.view(state.shape[0], -1)

        # encode current state and create latent space
        latent_space = self.activation(self.encoder(flattened))
        # decode current state
        outState = self.activation(self.decoderState(latent_space))
        # Reshape VECTOR -> IMAGE
        deflattened_reconstruction = outState.reshape(original_shape)

        # append action to latent space
        latent_space_action = torch.cat((latent_space, action), 1)

        # decode next state from latent space with action
        outNextState = self.activation(self.decoderNextState(latent_space_action))
        # Reshape VECTOR -> IMAGE
        deflattened_next_state = outNextState.reshape(original_shape)

        # decode difference between state and next state
        difference = self.activation(self.decoderDifference(latent_space_action))
        deflattened_difference = difference.reshape(original_shape)

        return deflattened_reconstruction, deflattened_next_state, deflattened_difference
