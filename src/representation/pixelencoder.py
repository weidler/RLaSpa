import numpy
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


class PixelEncoder(torch.nn.Module):

    def __init__(self):
        super(PixelEncoder, self).__init__()
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1, return_indices=True)

        # I tried, but this is not working.
        # Always getting error:
        #  in _unpool_output_size    kernel_size[d] - 2 * padding[d])
        # IndexError: tuple index out of range

        # self.encoder = torch.nn.Linear(121, 50)
        # self.decoder = torch.nn.Linear(50, 121)

        self.unpool = torch.nn.MaxUnpool2d(kernel_size=2, stride=2, padding=1)
        self.unconv = torch.nn.ConvTranspose2d(1, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, input_tensor):
        out = self.conv(input_tensor)
        pooled, indices = self.pool(out)

        # encoded = self.encoder(pooled.view(1, 121))
        # decoded = self.decoder(encoded)

        unpooled = self.unpool(pooled, indices)
        unconved = self.unconv(unpooled)

        return unconved



if __name__ == "__main__":

    img = torch.from_numpy(np.eye(20))
    img = img.view(1, 1, 20, 20).float()

    net = PixelEncoder()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.05)

    for epoch in range(10000):
        optimizer.zero_grad()

        out = net(img)
        loss = criterion(out, img)

        loss.backward()
        optimizer.step()

    print(np.round(out.detach().numpy()))


