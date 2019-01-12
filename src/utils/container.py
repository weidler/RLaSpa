from torch import Tensor


class SARSTuple:
    """
    Tuple that contains state, action, reward, next_state data.
    """

    def __init__(self, state: Tensor, action: Tensor, reward: float, next_state: Tensor):
        """
        Creates a state, action, reward, next_state tuple.
        :param state: previous state of the environment.
        :param action: action taken by the agent.
        :param reward: reward received after acting.
        :param next_state: state of the environment after acting.
        """
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state

    def ordered_tuple(self) -> tuple:
        return self.state, self.action, self.reward, self.next_state
