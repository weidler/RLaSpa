from typing import Tuple


class SARSTuple():
    def __init__(self, state, action, reward, next_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state

    def ordered_tuple(self) -> tuple:
        return (self.state, self.action, self.reward, self.next_state)
