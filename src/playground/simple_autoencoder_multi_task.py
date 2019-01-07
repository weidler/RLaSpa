import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils

from src.representation.network.autoencoder import AutoencoderNetwork


# pass in numpy array (hence pass by reference)
# TODO: currently this generates warnings (div by zero) for zero-padded features
def normalize(mat):
    mat = (mat - np.mean(mat, 0)) / np.std(mat, 0)
    return mat


if __name__ == "__main__":
    tasks = ['cartpole', 'mountain_car']
    data = {}

    max_feat_dim = 0

    total_num_instance = 0

    for task in tasks:
        with open("../../data/" + task + ".pkl", "rb") as f:
            data[task] = pickle.load(f)

        print("Finished reading .pkl for", task)
        assert len(data[task]) != 0

        max_feat_dim = max(len(data[task][0][0]), max_feat_dim)
        total_num_instance += len(data[task])

    for task_idx, task in enumerate(tasks):
        # normalize input to mean=0, variance=1
        # currently it's done not so efficiently to avoid changes in other parts of code
        normalized_input = np.zeros((len(data[task]), max_feat_dim))
        for i, instance in enumerate(data[task]):
            normalized_input[i, :len(instance[0])] = np.array(instance[0])

        # normalized_input = normalize(normalized_input)

        for i, instance in enumerate(data[task]):  # write back to the l
            data[task][i][0] = normalized_input[i]

        print("Finished normalizing features for", task)

        print('Creating input tensor for', task)
        # append this to get task index in task dict: list(task_dict.keys()).index(task)
        data[task] = torch.stack([torch.Tensor(i[0]) for i in data[task]])

    tensor_x = torch.cat((data['cartpole'], data['mountain_car']), 0)
    dataset = utils.TensorDataset(tensor_x, tensor_x)  # create dataset
    train_loader = utils.DataLoader(dataset, batch_size=32, shuffle=True)

    net = AutoencoderNetwork(input_neurons=4, hidden_neurons=3, output_neurons=4)
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('epoch %d, instance %5d, loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    torch.save(net.state_dict(), "../../models/very-simple-multi-task.model")
