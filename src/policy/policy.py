import abc

class _Policy(abc.ABC):

    @abc.abstractmethod
    def update(self, state, action, reward, next_state) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def choose_action(self, state) -> int:
        raise NotImplementedError
