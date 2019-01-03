import abc


class _Policy(abc.ABC):

    @abc.abstractmethod
    def update(self, state, action, reward, next_state, next_action=None) -> None:
        """ Update Policy/Q-values based on observations from environment.

        :param state:
        :param action:
        :param reward:
        :param next_state:
        """
        raise NotImplementedError

    @abc.abstractmethod
    def choose_action(self, state) -> int:
        """ Choose an action for the given state based on current policy.

        :param state:
        """
        raise NotImplementedError
