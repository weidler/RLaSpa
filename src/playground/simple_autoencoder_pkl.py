import pickle
import random

import torch
import torch.nn as nn
import torch.optim as optim

from src.representation.network.autoencoder import AutoencoderNetwork

if __name__ == "__main__":
    with open("../../data/cartpole.pkl", "rb") as f:
        data = pickle.load(f)
    print("READ FILE")

    # normalize input to mean=0, variance=1
    # currently it's done not so efficiently to avoid changes in other parts of code
    # assert len(data) != 0
    # normalized_input = np.zeros((len(data), len(data[0][0])))  # shape is num_sinstances *  num_features
    #
    # for i, instance in enumerate(data):
    #     normalized_input[i] = np.array(instance[0])
    # m = np.mean(normalized_input, 0)
    # std = np.std(normalized_input, 0)
    # normalized_input = (normalized_input - m) / std
    #
    # for i in np.arange(len(data)):
    #     data[i][0] = normalized_input[i]
    # print('Finished normalization')

    net = AutoencoderNetwork()
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
