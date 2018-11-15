import random

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Linear(4, 3)
        self.decoder = nn.Linear(3, 4)

        self.activation = torch.sigmoid

    def forward(self, vinput):
        out = self.activation(self.encoder(vinput))
        out = (self.decoder(out))

        return out


if __name__ == "__main__":
    with open("../../data/cartpole.data") as f:
        data = [list(map(eval, l[:-1].split("\t"))) for l in f.readlines()]
    print("READ FILE")

    net = Autoencoder()
    optimizer = optim.SGD(net.parameters(), lr=0.1)
    criterion = nn.MSELoss()

    for epoch in range(10):
        for i in range(10000):
            sample_id = random.randint(0, len(data) - 1)
            vinput = torch.tensor(data[sample_id][0])  # + [data[sample_id][1]])
            vtarget = torch.tensor(data[sample_id][0])

            optimizer.zero_grad()
            out = net(vinput)
            loss = criterion(out, vtarget)
            loss.backward()
            optimizer.step()
        if (epoch % 1 == 0): print(epoch)

    torch.save(net.state_dict(), "../../models/very-simple.model")
