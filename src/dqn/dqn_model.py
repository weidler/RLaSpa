import random

import torch
import torch.nn as nn
import torch.nn.functional as f


class DQN(nn.Module):
    def __init__(self, num_features, num_actions, gamma: float):
        super(DQN, self).__init__()
        self.num_feature = num_features
        self.num_actions = num_actions
        self.layer1 = nn.Linear(num_features, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, num_actions)
        self.gamma = gamma

    def forward(self, x):
        """
        Network forward pass.

        :param x: network input
        :return: network output
        """
        x = f.relu(self.layer1(x))
        x = f.relu(self.layer2(x))
        return self.layer3(x)

    def act(self, state, epsilon: float) -> int:
        """
        Method that returns the action the agent will do. This method uses the iteration value
        to calculate epsilon and choose between exploration and exploitation.

        :param state: actual state
        :param epsilon: exploration factor
        :return: the action that the agent will take
        """
        if random.random() > epsilon:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_value = self.forward(state)
            action = torch.argmax(q_value).item()
        else:
            action = random.randrange(self.num_actions)
        return action
