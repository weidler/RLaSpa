import abc
import random

from src.utils.container import SARSTuple


class _RepresentationLearner(abc.ABC):

    @abc.abstractmethod
    def encode(self, state):
        raise NotImplementedError

    @abc.abstractmethod
    def learn(self, state, action, reward, next_state, remember=True):
        raise NotImplementedError

    def learn_many(self, sars_tuples, remember=True):
        total_loss = 0
        sample: SARSTuple
        for sample in sars_tuples:
            total_loss += self.learn(*sample.ordered_tuple(), remember=remember)

        return total_loss / len(sars_tuples)

    def learn_from_backup(self):
        random.shuffle(self.backup_history)
        self.learn_many(self.backup_history, remember=False)

    def clear_backup(self):
        self.backup_history = []
