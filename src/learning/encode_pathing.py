import copy
import random

import numpy as np
import torch

from src.representation.visual.pixelencoder import SimplePixelEncoder, SiamesePixelEncoder
from src.task.pathing import ObstaclePathing, SimplePathing

env = ObstaclePathing(30, 30,
                          [[0, 13, 18, 20],
                           [16, 18, 11, 30],
                           [0, 25, 6, 8]]
                          )
# env = SimplePathing(10, 10)

use_history = True  # currently there is no going back, need to adjust non-history mode first
if use_history:
    with open("../../data/pathing_history.his") as f:
        data = [list(map(eval, l[:-1].split("\t"))) for l in f.readlines()]

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


net = SiamesePixelEncoder(env.width, env.height, 4, 50)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
epochs = 100000  # in a 10 by 10 simple env it needs at least around 800,000 epochs to have an idea where the
# agent is currently at if he is moving

for epoch in range(epochs):
    # new sample
    if not use_history:
        frames = make_steps(3)
        dynamic_view = combine_steps(frames)
        img = get_tensor(dynamic_view)
    else:
        sample_id = random.randint(0, len(data) - 2)
        state, action = data[sample_id]
        action = [1 if a == action else 0 for a in range(4)]
        action = torch.tensor([action]).float()
        next_state, _ = data[sample_id + 1]

        env.current_state = state
        img = get_tensor(env.get_pixelbased_representation())
        env.current_state = next_state
        img_next = get_tensor(env.get_pixelbased_representation())

    optimizer.zero_grad()

    reconstruction, next_state_prediction = net(img, action)
    loss_rec = criterion(reconstruction, img)
    loss_next = criterion(next_state_prediction, img_next)
    loss = sum((loss_rec, loss_next))
    loss.backward()
    optimizer.step()

    if epoch % (epochs / 100) == 0:
        print("{2}%; Epoch: {0}/{1}".format(epoch, epochs, round(epoch / epochs * 100, 0)))

    if epoch % (epochs / 10) == 0:
        env.visualize(img=reconstruction.tolist()[0][0])
        env.visualize(img=next_state_prediction.tolist()[0][0])

out = net(img)
env.visualize(img=out.tolist()[0][0])
