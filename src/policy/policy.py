import abc


class _Policy(abc.ABC):

    @abc.abstractmethod
    def update(self, state, action, reward, next_state, done) -> None:
        """
        Update Policy/Q-values based on observations from environment.

        :param state: original environment state
        :param action: action done by the agent
        :param reward: reward received
        :param next_state: environment state after acting
        :param done: environment finished after acting
        """
        raise NotImplementedError

    @abc.abstractmethod
    def choose_action(self, state) -> int:
        """
        Choose an action for the given state based on current policy. The iteration number is included
        to use it for exploration/exploitation decisions.

        :param state: current state of the environment
        """
        raise NotImplementedError

    @abc.abstractmethod
    def choose_action_policy(self, state) -> int:
        """
        Choose an action for the given state based on current policy. No exploration is considered in this
        method. Used for testing the training policy

        :param state: current state of the environment
        """
        raise NotImplementedError

    @abc.abstractmethod
    def finish_training(self) -> None:
        """
        Method that executes the agent necessary routine (if needed) after finishing the training
        """
        raise NotImplementedError
