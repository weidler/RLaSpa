import abc


class _RepresentationLearner(abc.ABC):

    @abc.abstractmethod
    def encode(self, state):
        raise NotImplementedError

    @abc.abstractmethod
    def learn(self, state):
        raise NotImplementedError

    @abc.abstractmethod
    def learn_many(self, states):
        raise NotImplementedError

    @abc.abstractmethod
    def learn_from_backup(self):
        raise NotImplementedError
