import copy
import random

import numpy as np
import torch

from src.representation.visual.pixelencoder import SimplePixelEncoder
from src.task.pathing import ObstaclePathing, SimplePathing

env = ObstaclePathing(30, 30, [[4, 9, 3, 8], [14, 19, 20, 26]])
env = SimplePathing(10, 10)

def make_steps(n=10) -> list:
    frames = [env.get_pixelbased_representation()]
    for step in range(n):
        env.step(random.randint(0, 3))
        frames.append(env.get_pixelbased_representation())

    return frames


def combine_steps(frames: list) -> list:
    return list(np.maximum.reduce([np.matrix(f) for f in frames]).tolist())


def get_tensor(view: list):
    img = torch.from_numpy(np.matrix(view))
    img = img.view(1, 1, env.width, env.height).float()

    return img


frames = make_steps(3)
dynamic_view = combine_steps(frames)
img = get_tensor(dynamic_view)
env.visualize(dynamic_view)

net = SimplePixelEncoder(env.width, env.height, 50)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
epochs = 1000000  # in a 10 by 10 simple env it needs at least around 800,000 epochs to have an idea where the
# agent is currently at if he is moving

for epoch in range(epochs):
    # new sample
    frames = make_steps(3)
    dynamic_view = combine_steps(frames)
    img = get_tensor(dynamic_view)

    optimizer.zero_grad()

    out = net(img)
    loss = criterion(out, img)

    loss.backward()
    optimizer.step()

    if epoch % (epochs / 100) == 0:
        print("{2}%; Epoch: {0}/{1}".format(epoch, epochs, round(epoch / epochs * 100, 0)))

    if epoch % (epochs / 10) == 0:
        env.visualize(img=out.tolist()[0][0])

out = net(img)
env.visualize(img=out.tolist()[0][0])
