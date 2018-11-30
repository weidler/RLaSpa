import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.replay_memory import ReplayMemory


class DQN(nn.Module):
    def __init__(self, num_features, num_actions, init_epsilon: float, min_epsilon: float, epsilon_decay: int,
                 gamma: float, memory_size: int):
        super(DQN, self).__init__()
        self.num_feature = num_features
        self.num_actions = num_actions
        self.layer1 = nn.Linear(num_features, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, num_actions)
        self.init_epsilon = init_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.replay_memory = ReplayMemory(memory_size)

    def forward(self, x):
        """
        Network forward pass.

        :param x: network input
        :return: network output
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_value = self.forward(state)
            action = torch.argmax(q_value).item()
        else:
            action = random.randrange(self.num_actions)
        return action

    def calculate_epsilon(self, iteration):
        """
        Method that calculate epsilon depending of the training iteration number. It converges to
        min_epsilon

        :param iteration: iteration number
        :return: epsilon for the iteration
        """
        return self.min_epsilon + (self.init_epsilon - self.min_epsilon) * math.exp(
            -1. * iteration / self.epsilon_decay)
