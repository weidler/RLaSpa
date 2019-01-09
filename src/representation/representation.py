import abc
import random
import torch
from typing import List

from src.utils.container import SARSTuple


class _RepresentationLearner(abc.ABC):

    @abc.abstractmethod
    def encode(self, state) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def learn(self, state, action, reward, next_state, remember=True) -> float:
        raise NotImplementedError

    def learn_many(self, sars_tuples: List[SARSTuple], remember=True) -> float:
        total_loss = 0
        for sample in sars_tuples:
            total_loss += self.learn([sample.state], [sample.action], [sample.reward], [sample.next_state], remember=remember)

        return total_loss / len(sars_tuples)

    def learn_from_backup(self) -> None:
        random.shuffle(self.backup_history)
        self.learn_many(self.backup_history, remember=False)

    def clear_backup(self) -> None:
        self.backup_history = []
