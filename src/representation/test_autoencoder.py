import pickle
import random

import torch
from torch import nn

from src.representation.autoencoder import Autoencoder

net = Autoencoder()
#net.load_state_dict(torch.load('../../models/hidden-2-ae.model'))
net.load_state_dict(torch.load('../../models/very-simple.model'))

criterion = nn.MSELoss()

with open("../../data/cartpole.pkl", "rb") as f:
    data = pickle.load(f)

errors = []

for i in range(100):  # test 100 random inputs
    random_index = random.randint(0, len(data)-1)
    input_net = torch.tensor(data[random_index][0]).float()
    output_net = net.forward(input_net)
    print("The input was: {0}.\n The output was: {1}".format(input_net, output_net))
    errors.append(criterion.forward(output_net, input_net))

errors = torch.stack(errors)
print("The mean of the errors is {0}".format(errors.mean()))
