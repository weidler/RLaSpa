import torch

import numpy as np

from src.policy.policy import _Policy
from src.utils.exploration import boltzmann_explore

# CLASSES

class QTableOffPolicy(_Policy):

    def __init__(self, features: list, n_actions: int, learning_rate=1, gamma=0.999, temperature=3):
        """
        Q-Table approach learning off-policy. Default learning rate is 1 and therefore applies to
        deterministic environments.

        :param features: List of features given by their bin sizes.
        :param n_actions: Number of actions in the modeled environment.
        :param learning_rate: Balancing weight for old and new q values.
        :param gamma: Discount factor. Default: 0.9.
        :param temperature: Parameter used in the selection of an action
        """
        self.features = features
        self.q_table = torch.zeros(features + [n_actions])
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.temperature = temperature  # TODO: Maybe this should decrease over time using an Schedule?

    def update(self, state, action, reward, next_state, done):
        state_value = max(self._get_entry(next_state))
        old_q_value = self._get_entry(state)[action]
        new_q_value = old_q_value * (1 - self.learning_rate) + self.learning_rate * (reward + self.gamma * state_value)
        self._change_entry_at(torch.cat((state, torch.Tensor([action]))), new_q_value)

        return 0

    def choose_action(self, state):
        entry = self._get_entry(state)
        action = boltzmann_explore(entry, self.temperature)  # Here depending of the iteration calculate the temperature
        return action

    def choose_action_policy(self, state):
        entry = self._get_entry(state)
        action = np.argmax(entry)
        return action

    def _get_entry(self, state: list):
        indices = [i.long().view([1 for d in range(depth)] + [-1]) for depth, i in enumerate(state)]
        return self.q_table[indices].squeeze()

    def _change_entry_at(self, indices, new_value):
        indices = [i.long().view([1 for d in range(depth)] + [-1]) for depth, i in enumerate(indices)]
        self.q_table[indices] = new_value

    def finish_training(self) -> None:
        pass

    def restore_from_state(self, input) -> None:
        pass

    def get_current_training_state(self):
        pass


class QTableSARSA(QTableOffPolicy):

    def __init__(self, features: list, n_actions: int, learning_rate=1, gamma=0.99, temperature=2.0):
        """
        Q-Table approach working on-policy, called the SARSA algorithm.

        :param features: List of features given by their bin sizes.
        :param n_actions: Number of actions in the modeled environment.
        :param learning_rate: Balancing weight for old and new q values.
        :param gamma: Discount factor. Default: 0.9.
        :param temperature: Parameter used in the selection of an action
        """
        super().__init__(features, n_actions, learning_rate, gamma, temperature)

    def update(self, state, action, reward, next_state, done):
        next_action = self.choose_action_policy(next_state)
        state_value = self._get_entry(next_state)[next_action]  # on policy
        state_q_values = self._get_entry(state)
        old_q_value = state_q_values[action]
        new_q_value = old_q_value * (1 - self.learning_rate) + self.learning_rate * (reward + self.gamma * state_value)
        self.q_table = self._change_entry_at(state + [action], new_q_value)
