import numpy

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


class PixelEncoder(torch.nn.Module):

    def __init__(self):
        super(PixelEncoder, self).__init__()
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)

        self.unpool = torch.nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.unconv = torch.nn.ConvTranspose2d(1, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, input_tensor):
        out = self.conv(input_tensor)
        pooled, indices = self.pool(out)
        unpooled = self.unpool(pooled, indices)
        unconved = self.unconv(unpooled)

        return unconved



if __name__ == "__main__":

    img = torch.from_numpy(numpy.eye(10))
    img = img.view(1, 1, 10, 10).float()

    net = PixelEncoder()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.05)

    for epoch in range(10000):
        optimizer.zero_grad()

        out = net(img)
        loss = criterion(out, img)
        loss.backward()
        optimizer.step()

    print(out.detach().numpy())

