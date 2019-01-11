import abc
import random
import torch
from torch import Tensor
from typing import List

from src.utils.container import SARSTuple


class _RepresentationLearner(abc.ABC):

    @abc.abstractmethod
    def __init__(self):
        self.network = None  # placeholder in superclass, for the convenience of saving/loading
        self.optimizer = None

    @abc.abstractmethod
    def encode(self, state: Tensor) -> Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def learn(self, state: Tensor, action: Tensor, reward: Tensor, next_state: Tensor, remember: bool = True) -> float:
        raise NotImplementedError

    def learn_batch_of_tuples(self, batch: List[SARSTuple]):
        # batch_size = len(batch)
        # state_size = batch[0].state.shape
        # action_size = batch[0].action.shape

        state_batch = torch.stack([s.state for s in batch], 0)
        action_batch = torch.stack([s.action for s in batch], 0)
        reward_batch = torch.Tensor([s.reward for s in batch])
        next_state_batch = torch.stack([s.next_state for s in batch], 0)

        # state_batch = torch.zeros((batch_size,) + state_size)
        # action_batch = torch.zeros((batch_size,) + action_size)
        # reward_batch = torch.zeros((batch_size, 1))
        # next_state_batch = torch.zeros((batch_size,) + state_size)
        # for i, sars_tuple in enumerate(batch):
        #     state_batch.append(sars_tuple.state)
        #     action_batch.append(sars_tuple.action)
        #     reward_batch.append(sars_tuple.reward)
        #     next_state_batch.append(sars_tuple.next_state)

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
