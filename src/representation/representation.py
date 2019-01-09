import abc
import random
from typing import List

from src.utils.container import SARSTuple


class _RepresentationLearner(abc.ABC):

    @abc.abstractmethod
    def encode(self, state):
        raise NotImplementedError

    @abc.abstractmethod
    def learn(self, state, action, reward, next_state, remember=True):
        raise NotImplementedError

    def learn_batch_of_tuples(self, batch: List[SARSTuple]):
        state_batch, action_batch, reward_batch, next_state_batch = [], [], [], []
        for sars_tuple in batch:
            state_batch.append(sars_tuple.state)
            action_batch.append(sars_tuple.action)
            reward_batch.append(sars_tuple.reward)
            next_state_batch.append(sars_tuple.next_state)

        # learn
        self.learn(
            state=state_batch,
            action=action_batch,
            reward=reward_batch,
            next_state=next_state_batch
        )
