import random

import numpy as np
import torch

from src.task.pathing import ObstaclePathing, SimplePathing


class PixelEncoder(torch.nn.Module):

    def __init__(self):
        super(PixelEncoder, self).__init__()
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1, return_indices=True)

        self.encoder = torch.nn.Linear(225, 50)
        self.decoder = torch.nn.Linear(50, 225)

        self.unpool = torch.nn.MaxUnpool2d(kernel_size=2, stride=2, padding=1)
        self.unconv = torch.nn.ConvTranspose2d(1, 1, kernel_size=3, stride=1, padding=0)

    def forward(self, input_tensor):
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

    def forward(self, input_tensor):
        original_shape = input_tensor.shape

        # Reshape IMAGE -> VECTOR
        flattened = input_tensor.view(1, -1)

        encoded = self.encoder(flattened)
        decoded = self.decoder(encoded)

        # Reshape VECTOR -> IMAGE
        deflattened = decoded.reshape(original_shape)

        return deflattened


if __name__ == "__main__":
    pass