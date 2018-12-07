import random

import torch
import torch.nn as nn
import torch.optim as optim

from src.representation.autoencoder import Autoencoder

if __name__ == "__main__":
    with open("../../data/cartpole.data") as f:
        data = [list(map(eval, l[:-1].split("\t"))) for l in f.readlines()]
    print("READ FILE")

    # print(data[0])
    net = Autoencoder(5, 3, 4)
    optimizer = optim.SGD(net.parameters(), lr=0.1)
    criterion = nn.MSELoss()

    for epoch in range(10):
        for i in range(10000):
            sample_id = random.randint(0, len(data) - 1)
            vinput = torch.tensor(data[sample_id][0] + [data[sample_id][1]])
            vtarget = torch.tensor(data[sample_id + 1][0])

            optimizer.zero_grad()
            out = net(vinput)
            loss = criterion(out, vtarget)
            loss.backward()
            optimizer.step()
        if epoch % 1 == 0:
            print(epoch)

    torch.save(net.state_dict(), "../../models/sas.model")
