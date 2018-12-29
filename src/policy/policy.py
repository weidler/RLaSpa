import abc


class _Policy(abc.ABC):

    @abc.abstractmethod
    def update(self, state, action, reward, next_state, done, next_action=None) -> None:
        """ Update Policy/Q-values based on observations from environment.

        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done
        :param next_action:
        """
        raise NotImplementedError

    @abc.abstractmethod
    def choose_action(self, state, iteration: int) -> int:
        """
        Choose an action for the given state based on current policy. The iteration number is included
        to use it for exploration/exploitation decisions.

        :param state: current state of the environment
        :param iteration: iteration number in the game
        """
        raise NotImplementedError
