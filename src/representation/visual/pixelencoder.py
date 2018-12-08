import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from src.task.pathing import ObstaclePathing


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


if __name__ == "__main__":

    task = ObstaclePathing(30, 30, [[4, 9, 3, 8], [14, 19, 20, 26]])
    task.visualize()

    img = torch.from_numpy(np.matrix(task.get_pixelbased_representation()))
    img = img.view(1, 1, task.width, task.height).float()

    net = PixelEncoder()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
    epochs = 100000

    for epoch in range(epochs):
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
