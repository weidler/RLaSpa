import random
from collections import deque

import torch

from src.utils.memory.memory import Memory


class ReplayMemory(Memory):
    def __init__(self, capacity: int):
        """
        Create a replay memory.

        :param capacity: number of memories that will be saved
        """
        self.memory = deque(maxlen=capacity)

    def push(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor,
             done: torch.Tensor) -> None:
        """
        Method to save the plays made by the agent in the memory

        :param state: state of the game before executing the action
        :param action: action taken by the agent
        :param reward: reward received from the action
        :param next_state: state of the game after executing the action
        :param done: true if the game is finished after executing the action
        """
        state = state.unsqueeze(0)
        # transform to tensor
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)
        next_state = next_state.unsqueeze(0)
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """
        Method to obtain a sample of saved memories

        :param batch_size: number of memories to retrieve
        :return: batch of memories
        """
        state, action, reward, next_state, done = zip(*random.sample(self.memory, batch_size))
        return torch.cat(state).detach(), torch.stack(action), torch.stack(reward), torch.cat(next_state).detach(), torch.stack(done)

    def __len__(self):
        return len(self.memory)
