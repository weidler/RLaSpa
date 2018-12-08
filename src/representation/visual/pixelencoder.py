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
    env = ObstaclePathing(30, 30, [[4, 9, 3, 8], [14, 19, 20, 26]])
    env = SimplePathing(10, 10)
    env.visualize()


    def make_steps(n=10):
        for step in range(n):
            env.step(random.randint(0, 3))


    def get_tensor():
        img = torch.from_numpy(np.matrix(env.get_pixelbased_representation()))
        img = img.view(1, 1, env.width, env.height).float()

        return img


    net = SimplePixelEncoder(env.width, env.height, 50)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
    epochs = 1000000    # in a 10 by 10 simple env it needs at least around 800,000 epochs to have an idea where the
                        # agent is currently at if he is moving

    for epoch in range(epochs):
        # new sample
        make_steps(10)
        img = get_tensor()

        optimizer.zero_grad()

        out = net(img)
        loss = criterion(out, img)

        loss.backward()
        optimizer.step()

        if epoch % (epochs / 10) == 0:
            print("{2}%; Epoch: {0}/{1}".format(epoch, epochs, round(epoch / epochs * 100, 0)))
            env.visualize(img=out.tolist()[0][0])

    out = net(img)
    env.visualize(img=out.tolist()[0][0])
