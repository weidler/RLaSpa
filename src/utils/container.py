from typing import Tuple

import torch
from torch import Tensor


class SARSTuple():
    next_state: Tensor
    reward: Tensor
    action: Tensor
    state: Tensor

    def __init__(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state

    def ordered_tuple(self) -> tuple:
        return self.state, self.action, self.reward, self.next_state
