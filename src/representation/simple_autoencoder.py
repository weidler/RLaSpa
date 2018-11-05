import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Linear(8, 3)
        self.decoder = nn.Linear(3, 8)

        self.activation = torch.sigmoid

    def forward(self, vinput):
        out = self.activation(self.encoder(vinput))
        out = self.activation(self.decoder(out))

        return out


if __name__ == "__main__":
    net = Autoencoder()
    optimizer = optim.SGD(net.parameters(), lr=0.999)
    criterion = nn.MSELoss()

    data = [torch.Tensor([0 if j != i else 1 for j in range(8)]).view(1, -1) for i in range(8)]
    print(data)

    for epoch in range(10000):
        for sample in data:
            optimizer.zero_grad()
            out = net(sample)
            loss = criterion(out, sample)
            loss.backward()
            optimizer.step()

    for sample in data:
        print(sample)
        print(net(sample))
