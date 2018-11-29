import random
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class DQN(nn.Module):
    def __init__(self, num_features, num_action):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(num_features, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, num_action)

    def forward(self, x):
        """
        Network forward pass.

        :param x: network input
        :return: network output
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

    def act(self, state, epsilon, num_actions):
        if random.random() > epsilon:
            state = torch.tensor(state, ndtype=torch.float32).unsqueeze(0)
            q_value = self.forward(state)
            action = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(num_actions)
        return action
