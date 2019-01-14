import random

import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, num_features: int, num_actions: int, representation_network: torch.nn.Module):
        """
        Initializes the neural network that will predict the q value of the states

        :param num_features: number of features in the state
        :param num_actions: number of actions that the agent can do
        :param representation_network: Optional nn.Module used for the representation. Including it into the policy
        network allows full backpropagation.
        """
        super(DQN, self).__init__()
        self.num_features = num_features
        self.num_actions = num_actions
        # Network structure
        self.representation_network = representation_network
        self.layers = nn.Sequential(
            nn.Linear(self.num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions)
        )

    def forward(self, x):
        """
        Network forward pass.

        :param x: network input
        :return: network output
        """
        return self.layers(x)

    def act(self, state: torch.Tensor, epsilon: float) -> int:
        """
        Method that returns the action the agent will do. This method uses the iteration value
        to calculate epsilon and choose between exploration and exploitation.

        :param state: actual state
        :param epsilon: exploration factor
        :return: the action that the agent will take
        """
        if random.random() > epsilon:
            state = state.unsqueeze(0)
            q_value = self.forward(state)
            action = torch.argmax(q_value).item()
        else:
            action = random.randrange(self.num_actions)
        return action
