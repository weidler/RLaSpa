import pickle
import random

import torch
from torch import nn

from autoencoder import Autoencoder

net = Autoencoder(5, 3, 4)
#net.load_state_dict(torch.load('../../models/hidden-2-ae.model'))
net.load_state_dict(torch.load('../../models/sas.model'))

criterion = nn.MSELoss()

with open("../../data/cartpole.data") as f:
    data = [list(map(eval, l[:-1].split("\t"))) for l in f.readlines()]

errors = []

for i in range(100):  # test 100 random inputs
    # test predicting next state given current one + action
    random_index = random.randint(0, len(data)-2)
    input_net = torch.tensor(data[random_index][0] + [data[random_index][1]]).float()
    output_net = net.forward(input_net)
    print("The input was: {0}.\n The output was: {1}".format(input_net, output_net))
    errors.append(criterion.forward(output_net, torch.tensor(data[random_index + 1][0])))

errors = torch.stack(errors)
print("The mean of the errors is {0}".format(errors.mean()))
