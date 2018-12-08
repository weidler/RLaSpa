import random

import numpy as np

from src.utils.memory.memory import Memory
from src.utils.segment_tree import SumSegmentTree, MinSegmentTree


class PrioritizedReplayMemory(Memory):
    def __init__(self, capacity: int, alpha: float):
        """
        Create Prioritized Replay memory.

        :param capacity: max number of transitions to store in the buffer. When the memory overflows
            the old memories are dropped.
        :param alpha: how much prioritization is used (0 - no prioritization, 1 - full prioritization)
        """
        self._storage = []
        self._max_size = capacity
        self._next_idx = 0

        assert alpha > 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < capacity:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        """
        Method to save the plays made by the agent in the memory

        :param state: state of the game before executing the action
        :param action: action taken by the agent
        :param reward: reward received from the action
        :param next_state: state of the game after executing the action
        :param done: true if the game is finished after executing the action
        """
        idx = self._next_idx
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        data = (state, action, reward, next_state, done)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._max_size
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size: int, beta=1.0):
        """
        Sample a batch of experiences. It also returns importance weights and idxes of sampled experiences.

        :param batch_size: transitions to sample
        :param beta: degree to use importance weights (0 - no corrections, 1 - full correction)
        :return: a tuple with:
            state_batch: np.array
                batch of states
            act_batch: np.array
                batch of actions executed given obs_batch
            rew_batch: np.array
                rewards received as results of executing act_batch
            next_state_batch: np.array
                next set of states seen after executing act_batch
            done_mask: np.array
                done_mask[i] = 1 if executing act_batch[i] resulted in
                the end of an episode and 0 otherwise.
            weights: np.array
                Array of shape (batch_size,) and dtype np.float32
                denoting importance weight of each sampled transition
            idxes: np.array
                Array of shape (batch_size,) and dtype np.int32
                idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [idxes, weights])

    def update_priorities(self, idxes: [int], priorities: [float]):
        """
        Update priorities of sampled transitions, sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        :param idxes: List of idxes of sampled transitions
        :param priorities: List of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha
        self._max_priority = max(self._max_priority, priority)

    def _encode_sample(self, idxes: [int]):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in idxes:
            state, action, reward, next_state, done = self._storage[i]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        return np.concatenate(states), actions, rewards, np.concatenate(next_states), dones

    def __len__(self):
        return len(self._storage)
