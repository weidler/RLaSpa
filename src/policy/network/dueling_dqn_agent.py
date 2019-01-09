import random

import torch
import torch.nn as nn


class DuelingDQN(nn.Module):
    def __init__(self, num_features, num_actions):
        super(DuelingDQN, self).__init__()
        self.num_feature = num_features
        self.num_actions = num_actions
        self.feature = nn.Sequential(
            nn.Linear(self.num_feature, 128),
            nn.ReLU()
        )
        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions)
        )
        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        """
        Network forward pass.

        :param x: network input
        :return: network output
        """
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + (advantage - advantage.mean())

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
