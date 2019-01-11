import abc


class _Policy(abc.ABC):
    """ Abstract policy class. A policy provides an agent with a tool for making decisions about the best action to take
    in a current state of the environment. Each implementation of a policy has a method for updating the policy based
    on new observations, as well as methods that choose an action for a given state.
    """

    @abc.abstractmethod
    def update(self, state, action, reward, next_state, done) -> None:
        """ Update Policy/Q-values based on observations from environment.

        :param state:       original environment state
        :param action:      action done by the agent
        :param reward:      reward received
        :param next_state:  environment state after acting
        :param done:        environment finished after acting
        """
        raise NotImplementedError

    @abc.abstractmethod
    def choose_action(self, state) -> int:
        """ Choose an action for the given state based on current policy. The iteration number is included
        to use it for exploration/exploitation decisions.

        :param state:       current state of the environment
        """
        raise NotImplementedError

    @abc.abstractmethod
    def choose_action_policy(self, state) -> int:
        """ Choose an action for the given state based on current policy. No exploration is considered in this
        method. Used for testing the training policy

        :param state:       current state of the environment
        """
        raise NotImplementedError

    @abc.abstractmethod
    def finish_training(self) -> None:
        """ Executes the agents necessary routine (if needed) after finishing the training. """
        raise NotImplementedError

    @abc.abstractmethod
    def restore_from_state(self, state) -> None:
        """ Resume training from a dictionary specifying training status

        :param state:       dictionary specifying training status
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_current_training_state(self) -> dict:
        """ Save current training status in dictionary for resuming later. """
        raise NotImplementedError
