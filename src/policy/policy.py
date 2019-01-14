import abc

import torch


class _Policy(abc.ABC):
    """ Abstract policy class. A policy provides an agent with a tool for making decisions about the best action to take
    in a current state of the environment. Each implementation of a policy has a method for updating the policy based
    on new observations, as well as methods that choose an action for a given state.
    """

    @abc.abstractmethod
    def calculate_next_q_value(self, next_state: torch.Tensor) -> torch.Tensor:
        """
        Method that calculates the next Q value given the next state. Used to calculate the loss
        when the memory is not used. This method handles only one state.

        :param next_state: state of the environment after acting
        :return: estimation of the next state q value
        """
        raise NotImplementedError

    @abc.abstractmethod
    def calculate_next_q_value_memory(self, next_state: torch.Tensor) -> torch.Tensor:
        """
        Method that calculates the next Q value given the next state. Used to calculate the loss when
        the memory is in used. This method handle a list of states.

        :param next_state: list of states of the environment after acting
        :return: estimation of the next state q value
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compute_td_loss(self, state: torch.tensor, action: torch.tensor, reward: torch.tensor, next_state: torch.tensor,
                        done: torch.tensor) -> torch.tensor:
        """
        Method to compute the loss for a given iteration, in general is used when the memory mechanism is off

        :param state: initial state
        :param action: action taken
        :param reward: reward received
        :param next_state: state after acting
        :param done: flag that indicates if the episode has finished
        :return: loss tensor
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compute_td_loss_memory(self) -> torch.Tensor:
        """
        Method that computes the loss of a batch. The batch is sample for memory to take in consideration
        situations that happens before.

        :return: loss tensor
        """
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, state, action, reward, next_state, done) -> float:
        """ Update Policy/Q-values based on observations from environment.

        :param state:       original environment state
        :param action:      action done by the agent
        :param reward:      reward received
        :param next_state:  environment state after acting
        :param done:        environment finished after acting

        :return:            loss produced by this update
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
