import random
from typing import List

import gym
import torch
import src.gym_custom_tasks
from torch import Tensor

from src.agents.agent import reset_env, step_env
from src.policy.dqn import DoubleDeepQNetwork
from src.policy.policy import _Policy
from src.representation.learners import JanusPixel, CerberusPixel, CVAEPixel
from src.utils.model_handler import apply_checkpoint


class RandomPolicy(object):
    def __init__(self, num_actions: int):
        self.num_actions = num_actions

    def choose_action_policy(self, state) -> int:
        return random.randrange(self.num_actions)


class RepresentationViewer(object):
    def __init__(self, env_name: str, repr_learner_name: str, ckpt_path: str, load_policy: bool = True):
        # Activate CUDA if available
        if torch.cuda.is_available():
            print("Using GPU - Setting default tensor type to cuda.FloatTensor.")
            self.gpu = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            self.device = torch.device('cuda:0')
        else:
            self.gpu = False
            self.device = torch.device("cpu")
        # Load environments
        if env_name == 'tunnel':
            self.environments = [gym.make("Tunnel-v0")]
        elif env_name == 'scrollers':
            self.environments = [gym.make("Tunnel-v0"),
                                 gym.make("Evasion-v0"),
                                 gym.make("EvasionWalls-v0")
                                 ]
        elif env_name == 'pathing':
            self.environments = [gym.make("VisualObstaclePathing-v0"),
                                 gym.make("VisualObstaclePathing-v1"),
                                 gym.make("VisualObstaclePathing-v2"),
                                 gym.make("VisualObstaclePathing-v3")]
        else:
            raise ValueError('No such env {}'.format(env_name))
        # Create action vector
        self.width = self.environments[0].observation_space.shape[0]
        self.height = self.environments[0].observation_space.shape[1]
        self.num_actions = self.environments[0].action_space.n
        self.one_hot_actions = torch.eye(self.num_actions)
        # Load reprensentation learner
        if repr_learner_name == 'janus':
            self.representation_module = JanusPixel(width=self.width,
                                                    height=self.height,
                                                    n_actions=self.num_actions,
                                                    n_hidden=32)
        elif repr_learner_name == 'cerberus':
            self.representation_module = CerberusPixel(width=self.width,
                                                       height=self.height,
                                                       n_actions=self.num_actions,
                                                       n_hidden=32)
        elif repr_learner_name == 'cvae':
            self.representation_module = CVAEPixel(n_middle=32, n_hidden=16)
        else:
            raise ValueError('No such repr learner {}'.format(repr_learner_name))
        if load_policy:
            self.policy = DoubleDeepQNetwork(self.representation_module.n_hidden, self.num_actions, eps_calculator=None,
                                             memory_eps_calculator=None)
        else:
            self.policy = None
        steps_trained = apply_checkpoint(ckpt_path, policy=self.policy, repr=self.representation_module, gpu=self.gpu)
        self.representation_module.network.to(self.device)
        if load_policy:
            self.policy.model.to(self.device)
            self.policy.target_model.to(self.device)
        else:
            self.policy = RandomPolicy(num_actions=self.num_actions)
        print('Loaded representation learner {0} with {1} training steps.'.format(repr_learner_name, steps_trained))

    def get_representation(self, environment: gym.Env, number_of_snapshots: int,
                           steps_per_snapshot: int) -> List[Tensor]:
        state = reset_env(environment)
        steps = 0
        snapshots_taken = 0
        latent_representation = []
        while snapshots_taken < number_of_snapshots:
            action = self.policy.choose_action_policy(state)
            next_state, _, done, _ = step_env(action=action, env=environment)
            steps += 1
            if steps % steps_per_snapshot == 0:
                latent_representation.append(self.representation_module.encode(state))
                one_hot_action_vector = self.one_hot_actions[action]
                self.representation_module.visualize_output(state, one_hot_action_vector, next_state)
                snapshots_taken += 1
            if done:
                state = reset_env(environment)
            else:
                state = next_state
        return latent_representation


if __name__ == '__main__':
    env_name = 'scrollers'
    repr_learner_name = 'cerberus'
    ckpt_path = '/home/adrigrillo/Documents/RLaSpa/ckpt/ParallelAgent_Tunnel-v0_Evasion-v0_EvasionWalls-v0_CerberusPixel_DoubleDeepQNetwork/2019-01-21_02-04-24'

    # Load environments and representation learner
    visualizer = RepresentationViewer(env_name=env_name, repr_learner_name=repr_learner_name,
                                      ckpt_path=ckpt_path, load_policy=False)
    latent_representations = visualizer.get_representation(environment=visualizer.environments[2],
                                                           number_of_snapshots=20, steps_per_snapshot=5)
    print('done')

