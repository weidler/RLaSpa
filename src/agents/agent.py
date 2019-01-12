import abc
from typing import Tuple

import torch
from gym import Env
from torch import Tensor
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from src.policy.policy import _Policy
from src.representation.representation import _RepresentationLearner


class _Agent(abc.ABC):
    """ Abstract agent class. An agent unifies the three cornerstones of the system:
            - an environment in which the agent acts
            - a policy that it uses to make decisions about how to act
            - a representation module that converts an environment state into a latent representation.
        Implementations of the agent class provide methods for training the latter components for the purpose of acting
        in the environment.
    """
    representation_learner: _RepresentationLearner
    policy: _Policy
    env: Env

    @abc.abstractmethod
    def __init__(self, repr_learner: _RepresentationLearner, policy: _Policy, env: Env):
        self.env = env
        self.policy = policy
        self.representation_learner = repr_learner

    @abc.abstractmethod
    def train_agent(self, episodes: int, ckpt_to_load=None, save_ckpt_per=None, plot_every=None, log=False):
        """ Train the agent for some number of episodes. The max length of episodes is specified in the environment.
        Optionally save or load checkpoints from previous trainings.

        :param episodes:        the number of episodes
        :param ckpt_to_load:    (default None) loading checkpoint
        :param save_ckpt_per:   (default None) number of episodes after which a checkpoint is saved
        :param log:             (default False) whether logging is done
        """
        raise NotImplementedError

    def act(self, current_state: Tensor) -> Tuple[Tensor, float, bool]:
        latent_state = self.representation_learner.encode(current_state)
        action = self.policy.choose_action_policy(latent_state)
        next_state, step_reward, env_done, _ = self.step_env(action)

        return next_state, step_reward, env_done

    def step_env(self, action: int) -> Tuple[Tensor, float, bool, object]:
        """ Make a step in the environment and get the resulting state as a Tensor.

        :param action:  the action the agent is supposed to take in the environment
        """
        next_state, step_reward, env_done, info = self.env.step(action)
        tensor_state = torch.Tensor(next_state).float()

        return tensor_state, step_reward, env_done, info

    def reset_env(self) -> Tensor:
        """ Resets the environment and returns the starting state as a Tensor.

        :return:    the starting state
        """
        return torch.Tensor(self.env.reset()).float()

    def test(self, num_testruns=1, render=True) -> None:
        """ Run a test in the environment using the current policy without exploration. """
        all_rewards = []
        fig = plt.figure(figsize=(10, 6))
        for i in range(num_testruns):
            plt.clf()
            ims = []
            done = False
            state = self.reset_env()
            step = 0
            total_reward = 0
            while not done:
                state, reward, done = self.act(state)
                step += 1
                total_reward += reward
                ims.append([plt.imshow(state, cmap="binary", origin="upper", animated=True)])
                if render:
                    self.env.render()
            all_rewards.append(total_reward)
            print(f"Tested episode took {step} steps and gathered a reward of {total_reward}.")
            if not render:
                ani = animation.ArtistAnimation(fig, ims, blit=True,
                                                repeat_delay=1000)
                ani.save(f'../../data/testrun_{i}.gif', writer='imagemagick', fps=15)
        print(f'Average max score after {num_testruns} testruns: {sum(all_rewards)/len(all_rewards)}')

    def get_config_name(self):
        return "_".join(
            [self.__class__.__name__,
             self.env.__class__.__name__,
             self.representation_learner.__class__.__name__,
             self.policy.__class__.__name__])
