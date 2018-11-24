import pickle
import random

import torch
import torch.nn as nn
import torch.optim as optim

from src.representation.autoencoder import Autoencoder

if __name__ == "__main__":
    with open("../../data/cartpole.pkl", "rb") as f:
        data = pickle.load(f)
    print("READ FILE")

    net = Autoencoder()
    optimizer = optim.SGD(net.parameters(), lr=0.1)
    criterion = nn.MSELoss()

    for epoch in range(10):
        for i in range(10000):
            sample_id = random.randint(0, len(data) - 1)
            # Transform the tensor to float32
            vinput = torch.tensor(data[sample_id][0]).float()  # + [data[sample_id][1]])
            vtarget = torch.tensor(data[sample_id][0]).float()

            optimizer.zero_grad()
            out = net(vinput)
            loss = criterion(out, vtarget)
            loss.backward()
            optimizer.step()
        if epoch % 1 == 0:
            print(epoch)

    torch.save(net.state_dict(), "../../models/very-simple.model")
