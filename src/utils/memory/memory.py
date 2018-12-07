import abc


class Memory(abc.ABC):

    @abc.abstractmethod
    def push(self, state, action, reward, next_state, done):
        """
        Method to save the plays made by the agent in the memory

        :param state: state of the game before executing the action
        :param action: action taken by the agent
        :param reward: reward received from the action
        :param next_state: state of the game after executing the action
        :param done: true if the game is finished after executing the action
        """
        return NotImplementedError

    @abc.abstractmethod
    def sample(self, batch_size):
        """
        Method to obtain a sample of saved memories

        :param batch_size: number of memories to retrieve
        :return: batch of memories
        """
        return NotImplementedError