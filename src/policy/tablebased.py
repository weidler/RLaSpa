import numpy

from src.policy.policy import _Policy
from src.utils.exploration import boltzmann_explore


# FUNCTIONS

def _change_entry_at(table, indices, new_value):
    if len(indices) == 1:
        table[indices[0]] = new_value
        return table

    table[indices[0]] = _change_entry_at(table[indices[0]], indices[1:], new_value)
    return table


# CLASSES

class QTableOffPolicy(_Policy):

    def __init__(self, features: list, n_actions: int, learning_rate=1):
        """ Q-Table approach learning off-policy. Default learning rate is 1 and therefore applies to
        deterministic environments.

        :param features:        list of features given by their bin sizes
        :param n_actions:       number of actions in the modeled environment
        :param learning_rate:   balancing weight for old and new q values
        """
        self.features = features
        self.q_table = numpy.zeros(features + [n_actions])

        # HYPERPARAMETERS
        self.learning_rate = 1
        self.gamma = 0.99
        self.temperature = 2

    def update(self, state, action, reward, next_state, next_action=None):
        # update
        state_value = max(self._get_entry(next_state))
        old_q_value = self._get_entry(state)[action]
        new_q_value = old_q_value * (1 - self.learning_rate) + self.learning_rate * (reward + self.gamma * state_value)
        self.q_table = _change_entry_at(self.q_table, state + [action], new_q_value)

    def choose_action(self, state):
        # choose action with exploration
        entry = self._get_entry(state)
        action = boltzmann_explore(entry, self.temperature)

        return action

    def _get_entry(self, state):
        entry = self.q_table[state[0]]
        for ind in state[1:]:
            entry = entry[ind]
        return entry


class QTableSARSA(QTableOffPolicy):

    def __init__(self, features: list, n_actions: int, learning_rate=1):
        """ Q-Table approach working on-policy, called the SARSA  algorithm.

        :param features:
        :param n_actions:
        """
        super().__init__(features, n_actions)

    def update(self, state, action, reward, next_state, next_action=None):
        if next_action is None:
            raise ValueError("SARSA needs next action for on-policy calculations.")

        state_value = self._get_entry(next_state)[next_action]  # on policy
        old_q_value = self._get_entry(state)[action]
        new_q_value = old_q_value * (1 - self.learning_rate) + self.learning_rate * (reward + self.gamma * state_value)
        self.q_table = _change_entry_at(self.q_table, state + [action], new_q_value)
