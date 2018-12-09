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

class QTable(_Policy):

    def __init__(self, features: list, n_actions: int):
        """

        :param features:    list of features given by their bin sizes
        """
        self.features = features
        self.q_table = numpy.zeros(features + [n_actions])

        self.gamma = 0.99
        self.temperature = 2

    def update(self, state, action, reward, next_state):
        # update
        state_value = max(self._get_entry(next_state))
        entry = self.q_table[state[0]]
        for ind in state[1:]:
            entry = entry[ind]

        new_q_value = reward + self.gamma * state_value
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
