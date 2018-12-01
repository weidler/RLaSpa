import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.representation.autoencoder import Autoencoder

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

    for task in tasks:
        # normalize input to mean=0, variance=1
        # currently it's done not so efficiently to avoid changes in other parts of code
        normalized_input = np.zeros((len(data[task]), max_feat_dim))
        for i, instance in enumerate(data[task]):
            normalized_input[i, :len(instance[0])] = np.array(instance[0])

        normalized_input = normalize(normalized_input)

        for i, instance in enumerate(data[task]):  # write back to the l
            data[task][i][0] = normalized_input[i]
        print("Finished normalizing features for", task)


    # input = np.zeros((total_num_instance, max_feat_dim))  # shape is num_instances * highest num feats among all tasks

    net = Autoencoder()
    optimizer = optim.SGD(net.parameters(), lr=0.05)
    criterion = nn.MSELoss()

    for epoch in range(20):
        for i in range(10000):
            for task in tasks:  # alternate between tasks
                sample_id = random.randint(0, len(data) - 1)
                # Transform the tensor to float32
                vinput = torch.tensor(data[task][sample_id][0]).float()  # + [data[sample_id][1]])
                vtarget = torch.tensor(data[task][sample_id][0]).float()

                optimizer.zero_grad()
                out = net(vinput)
                loss = criterion(out, vtarget)
                loss.backward()
                optimizer.step()
        if epoch % 1 == 0:
            print(epoch)

    torch.save(net.state_dict(), "../../models/very-simple-multi-task.model")
