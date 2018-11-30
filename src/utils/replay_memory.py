import random
from collections import deque

import numpy as np


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Method to save the plays made by the agent in the memory

        :param state: state of the game before executing the action
        :param action: action taken by the agent
        :param reward: reward received from the action
        :param next_state: state of the game after executing the action
        :param done: true if the game is finished after executing the action
        """
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Method to obtain a sample of saved memories

        :param batch_size: number of memories to retrieve
        :return: batch of memories
        """
        state, action, reward, next_state, done = zip(*random.sample(self.memory, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.memory)
