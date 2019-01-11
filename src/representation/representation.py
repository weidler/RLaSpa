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

    def learn_batch_of_tuples(self, batch: List[SARSTuple]) -> None:
        state_batch = torch.stack([s.state for s in batch], 0)
        action_batch = torch.stack([s.action for s in batch], 0)
        reward_batch = torch.Tensor([s.reward for s in batch])
        next_state_batch = torch.stack([s.next_state for s in batch], 0)

        # learn
        self.learn(
            state=state_batch,
            action=action_batch,
            reward=reward_batch,
            next_state=next_state_batch
        )

    def current_state(self):
        # TODO: catch exception here, as flatten and so don't have network and optimizer
        return {
            'model': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

    def restore_from(self, restore_input):
        self.network.load_state_dict(restore_input['model'])
        self.optimizer.load_state_dict(restore_input['optimizer'])
