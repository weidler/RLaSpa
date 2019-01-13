from typing import Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
from gym import Env
from torch import Tensor

from src.policy.policy import _Policy
from src.representation.representation import _RepresentationLearner


class _Agent:
    """ Agent class. An agent unifies the three cornerstones of the system:
            - an environment in which the agent acts
            - a policy that it uses to make decisions about how to act
            - a representation module that converts an environment state into a latent representation.
        Implementations of the agent class provide methods for training the latter components for the purpose of acting
        in the environment.
    """

    def __init__(self, representation_learner: _RepresentationLearner, policy: _Policy, environment: Env):
        """
        Initializes the agent with the required elements.

        :param representation_learner: module that converts an environment state into a latent representation.
        :param policy: module that will make the decisions about how the agent is going to act.
        :param environment: environment in which the agent acts.
        """
        self.env = environment
        self.policy = policy
        self.representation_learner = representation_learner

    def train_agent(self, episodes: int, ckpt_to_load: str = None, episodes_per_saving: int = None,
                    plot_every: int = None, log: bool = False) -> None:
        """
        Train the agent for some number of episodes. The max length of episodes is specified in the environment.
        Optionally save or load checkpoints from previous trainings.

        :param episodes: the number of episodes
        :param ckpt_to_load: loading checkpoint. Default: None
        :param episodes_per_saving: number of episodes between saving checkpoint. Default: None
        :param plot_every: number of steps that will happen between the plotting of the space representation
        :param log: whether logging is done. Default: False
        """
        raise NotImplementedError

    def act(self, current_state: Tensor) -> Tuple[Tensor, float, bool]:
        """
        Method that makes the agent choose an action given the actual state. This method will imply the encoding
        of the state if a representation learner is capable of doing so.

        :param current_state: current state of the environment
        :return: next state of the environment along with the reward and a flag that indicates if
        the episode is finished
        """
        latent_state = self.representation_learner.encode(current_state)
        action = self.policy.choose_action_policy(latent_state)
        next_state, step_reward, env_done, _ = self.step_env(action)

        return next_state, step_reward, env_done

    def step_env(self, action: int) -> Tuple[Tensor, float, bool, object]:
        """
        Make a step in the environment and get the resulting state as a Tensor.

        :param action: the action the agent is supposed to take in the environment
        """
        next_state, step_reward, env_done, info = self.env.step(action)
        tensor_state = torch.Tensor(next_state).float()

        return tensor_state, step_reward, env_done, info

    def reset_env(self) -> Tensor:
        """
        Resets the environment and returns the starting state as a Tensor.

        :return: the starting state
        """
        return torch.Tensor(self.env.reset()).float()

    def test(self, runs_number=1, render=False) -> None:
        """
        Run a test in the environment using the current policy without exploration.

        :param runs_number: number of test to be done.
        :param render: render the environment
        """
        all_rewards = []
        fig = plt.figure(figsize=(10, 6))
        for i in range(runs_number):
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
                ani = animation.ArtistAnimation(fig, ims, blit=True, repeat_delay=1000)
                ani.save(f'../../data/testrun_{i}.gif', writer='imagemagick', fps=15)
        print(f'Average max score after {runs_number} testruns: {sum(all_rewards) / len(all_rewards)}')

    def get_config_name(self):
        return "_".join(
            [self.__class__.__name__,
             self.env.__class__.__name__,
             self.representation_learner.__class__.__name__,
             self.policy.__class__.__name__])
