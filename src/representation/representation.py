import abc
import torch
from torch import Tensor
from typing import List

from src.utils.container import SARSTuple


class _RepresentationLearner(abc.ABC):
    """ Abstract representation module class. A representation module is capable of learning to represent a given state
    in the latent space.
    """

    @abc.abstractmethod
    def __init__(self):
        self.network = None  # placeholder in superclass, for the convenience of saving/loading
        self.optimizer = None

    @abc.abstractmethod
    def encode(self, state: Tensor) -> Tensor:
        """ Encode the given state tensor into a latent representation.

        :param state:   the state to be encoded, given as a tensor
        """
        raise NotImplementedError

    @abc.abstractmethod
    def learn(self, state: Tensor, action: Tensor, reward: Tensor, next_state: Tensor) -> float:
        """ Learn from a presented (batch of) example(s). Learning examples are SARS tuples. Although not all
        implementations will need all four components, for consistency they need to be capable of getting passed each.

        :param state:       the current state of the environment when the action was taken
        :param action:      the action taken in the environment
        :param reward:      the reward received for taking the action in the state
        :param next_state:  the state observed after taking the action
        """
        raise NotImplementedError

    def learn_batch_of_tuples(self, batch: List[SARSTuple]) -> float:
        state_batch = torch.stack([s.state for s in batch], 0)
        action_batch = torch.stack([s.action for s in batch], 0)
        reward_batch = torch.Tensor([s.reward for s in batch])
        next_state_batch = torch.stack([s.next_state for s in batch], 0)

        # learn
        loss = self.learn(
            state=state_batch,
            action=action_batch,
            reward=reward_batch,
            next_state=next_state_batch
        )
        return loss

    def visualize_output(self, state: Tensor, action: Tensor, next_state: Tensor):
        """ Visualize some part of the representation learner.

        :param state:       state tensor
        :param action:      action tensor
        :param next_state:  next state tensor
        :return:
        """
        pass

    def current_state(self):
        """ Get current state of the representation learner
        Returns None if the repr learner does not contain a network (pass through, flatten)

        :return:            dict containing the network and optimizer
        """
        return {
            'model': None if self.network is None else self.network.state_dict(),
            'optimizer': None if self.network is None else self.optimizer.state_dict(),
        }

    def restore_from(self, restore_input):
        """ Restores the repr learner from some state

        :param restore_input:   dict containing the state of network and optimizer to restore
        :return:
        """
        if self.network is not None:
            self.network.load_state_dict(restore_input['model'])
            self.optimizer.load_state_dict(restore_input['optimizer'])
