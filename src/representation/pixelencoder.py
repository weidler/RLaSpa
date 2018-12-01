import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from src.task.ObstaclePathing import ObstaclePathing


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
        convoluted = self.conv(input_tensor)
        # pooled, indices = self.pool(out)

        # encoded = self.encoder(pooled.view(1, 121))
        # decoded = self.decoder(encoded)

        # unpooled = self.unpool(pooled, indices)
        unpooled = convoluted
        unconved = self.unconv(unpooled)

        return unconved


if __name__ == "__main__":

    task = ObstaclePathing(30, 30, [[4, 9, 3, 8], [14, 19, 20, 26]])
    task.visualize()

    img = torch.from_numpy(np.matrix(task.get_pixelbased_representation()))
    img = img.view(1, 1, task.width, task.height).float()

    net = PixelEncoder()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
    epochs = 100000

    for epoch in range(100000):
        optimizer.zero_grad()

        out = net(img)
        loss = criterion(out, img)

        loss.backward()
        optimizer.step()

        if epoch % (epochs / 10) == 0:
            print("{2}%; Epoch: {0}/{1}".format(epoch, epochs, round(epoch / epochs * 100, 0)))
            task.visualize(img=out.tolist()[0][0])

    out = net(img)
    task.visualize(img=out.tolist()[0][0])
