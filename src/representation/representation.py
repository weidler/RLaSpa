import abc


class _RepresentationLearner(abc.ABC):

    @abc.abstractmethod
    def encode(self, state):
        raise NotImplementedError

    @abc.abstractmethod
    def learn(self, state, action, reward, next_state, remember=True):
        raise NotImplementedError

    def learn_many(self, sars_tuples, remember=True):
        total_loss = 0
        for sample in sars_tuples:
            total_loss += self.learn(*sample, remember=remember)

        return total_loss / len(sars_tuples)

    def learn_from_backup(self):
        self.learn_many(self.backup_history, remember=False)
