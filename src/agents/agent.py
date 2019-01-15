import abc
from typing import Tuple, List
from typing import Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
from gym import Env
from torch import Tensor

from src.policy.policy import _Policy
from src.representation.representation import _RepresentationLearner
from src.utils.model_handler import save_checkpoint, apply_checkpoint
from src.utils.path_manager import PathManager
from src.utils.logger import Logger

def reset_env(env: Env) -> Tensor:
    """ Resets the environment and returns the starting state as a Tensor.

    :return:    the starting state
    """
    return torch.Tensor(env.reset()).float()


def step_env(action: int, env: Env) -> Tuple[Tensor, float, bool, object]:
    """ Make a step in the environment and get the resulting state as a Tensor.

    :param action:  the action the agent is supposed to take in the environment
    """
    next_state, step_reward, env_done, info = env.step(action)
    tensor_state = torch.Tensor(next_state).float()

    return tensor_state, step_reward, env_done, info


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
    environments: List[Env]

    @abc.abstractmethod
    def __init__(self, representation_learner: _RepresentationLearner, policy: _Policy, environments: List[Env]):
        self.environments = environments
        self.policy = policy
        self.representation_learner = representation_learner
        self.start_episode = 0
        self.path_manager = PathManager()
        self.logger = Logger('logs', self.get_config_name())

        # check if environments given as list
        if not isinstance(environments, list):
            raise ValueError("Need to provide list of environment. For single environment training provide single-element list.")

        # check if the environments are having same action space
        if not len(set([env.action_space.n for env in self.environments])) == 1:
            raise ValueError("All environments need to have the same amount of available actions!")

        # check if the environments are having same state space
        if not len(set([env.observation_space.shape for env in self.environments])) == 1:
            raise ValueError("All environments need to have the same state dimensionality!")


    @abc.abstractmethod
    def train_agent(self, episodes: int, ckpt_to_load: str = None, episodes_per_saving: int = None,
                    plot_every: int = None, log: bool = False) -> None:
        """ Train the agent for some number of episodes. The max length of episodes is specified in the environment.

        Optionally save or load checkpoints from previous trainings.

        :param episodes: the number of episodes
        :param ckpt_to_load: loading checkpoint. Default: None
        :param episodes_per_saving: number of episodes between saving checkpoint. Default: None
        :param plot_every: number of steps that will happen between the plotting of the space representation
        :param log: whether logging is done. Default: False
        """
        raise NotImplementedError

    def act(self, current_state: Tensor, env: Env) -> Tuple[Tensor, float, bool]:
        """
        Method that makes the agent choose an action given the actual state. This method will imply the encoding
        of the state if a representation learner is capable of doing so.

        :param current_state: current state of the environment
        :return: next state of the environment along with the reward and a flag that indicates if
        the episode is finished
        """

        latent_state = self.representation_learner.encode(current_state)
        action = self.policy.choose_action_policy(latent_state)
        next_state, step_reward, env_done, _ = step_env(action, env)

        return next_state, step_reward, env_done

    def test(self, env: Env, numb_runs: int = 1, render: bool = False) -> None:
        """
        Run a test in the environment using the current policy without exploration.

        :param numb_runs: number of test to be done.
        :param render: render the environment
        """

        all_rewards = []
        fig = plt.figure(figsize=(10, 6))
        for i in range(numb_runs):
            try:
                plt.clf()
                ims = []
                done = False
                state = reset_env(env)
                step = 0
                total_reward = 0
                while not done:
                    state, reward, done = self.act(state, env)
                    step += 1
                    total_reward += reward
                    ims.append([plt.imshow(state.cpu(), cmap="binary", origin="upper", animated=True)])
                    if render:
                        env.render()
                all_rewards.append(total_reward)
                print(f"Tested episode took {step} steps and gathered a reward of {total_reward}.")
                if not render:
                    ani = animation.ArtistAnimation(fig, ims, blit=True, repeat_delay=1000)
                    ani.save(f'../../data/{env.__class__.__name__}_testrun_{i}.gif', writer='imagemagick', fps=15)
            except Exception as e:
                print(f"Episode {i} went wrong: " + str(e))
        print(f'Average max score after {numb_runs} testruns: {sum(all_rewards) / len(all_rewards)}')

    def get_config_name(self):
        return "_".join(
            [self.__class__.__name__,
             "_".join([env.spec.id for env in self.environments]),
             self.representation_learner.__class__.__name__,
             self.policy.__class__.__name__])

    def save(self, episode: int, save_repr_learner: bool=True, save_policy_learner: bool=True) -> None:
        ckpt_dir = self.path_manager.get_ckpt_dir(self.get_config_name())

        if save_repr_learner:
            save_checkpoint(self.representation_learner.current_state(), episode, ckpt_dir, 'repr')

        if save_policy_learner:
            save_checkpoint(self.policy.get_current_training_state(), episode, ckpt_dir, 'policy')

    def load(self, ckpt_dir: str, load_repr_learner: bool=True, load_policy_learner: bool=True) -> None:

        if load_repr_learner and load_policy_learner:
            self.start_episode = apply_checkpoint(ckpt_dir, policy=self.policy, repr=self.representation_learner)

        elif load_repr_learner:
            self.start_episode = apply_checkpoint(ckpt_dir, repr=self.representation_learner)

        elif load_policy_learner:
            self.start_episode = apply_checkpoint(ckpt_dir, policy=self.policy)
