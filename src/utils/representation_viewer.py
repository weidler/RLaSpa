import random

import gym
import torch
import src.gym_custom_tasks
from torch import Tensor

from src.agents.agent import reset_env, step_env
from src.representation.learners import JanusPixel, CerberusPixel, CVAEPixel
from src.utils.model_handler import apply_checkpoint


class RepresentationViewer(object):
    def __init__(self, env_name: str, repr_learner_name: str, ckpt_path: str):
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
        self.num_actions = self.environments[0].action_space.n
        self.one_hot_actions = torch.eye(self.environments[0].action_space.n)
        # Load reprensentation learner
        if repr_learner_name == 'janus':
            self.representation_learner = JanusPixel(width=self.environments[0].observation_space.shape[0],
                                                     height=self.environments[0].observation_space.shape[1],
                                                     n_actions=self.environments[0].action_space.n,
                                                     n_hidden=32)
        elif repr_learner_name == 'cerberus':
            self.representation_learner = CerberusPixel(width=self.environments[0].observation_space.shape[0],
                                                        height=self.environments[0].observation_space.shape[1],
                                                        n_actions=self.environments[0].action_space.n,
                                                        n_hidden=32)
        elif repr_learner_name == 'cvae':
            self.representation_learner = CVAEPixel(n_middle=32, n_hidden=16)
        else:
            raise ValueError('No such repr learner {}'.format(repr_learner_name))
        steps_trained = apply_checkpoint(ckpt_path, repr=self.representation_learner, gpu=self.gpu)
        self.representation_learner.network.to(self.device)
        print('Loaded representation learner {0} with {1} training steps.'.format(repr_learner_name, steps_trained))

    def get_representation(self, environment: gym.Env) -> Tensor:
        state = reset_env(environment)
        action = random.randrange(self.num_actions)
        next_state, _, _, _ = step_env(action=action, env=environment)
        latent_space = self.representation_learner.encode(state)
        one_hot_action_vector = self.one_hot_actions[action]
        self.representation_learner.visualize_output(state, one_hot_action_vector, next_state)
        return latent_space


if __name__ == '__main__':
    env_name = 'tunnel'
    repr_learner_name = 'cerberus'
    ckpt_path = '/home/adrigrillo/Documents/RLaSpa/ckpt/ParallelAgent_Tunnel-v0_CerberusPixel_DoubleDeepQNetwork/' \
                + '2019-01-21_13-01-40'

    # Load environments and representation learner
    visualizer = RepresentationViewer(env_name=env_name, repr_learner_name=repr_learner_name, ckpt_path=ckpt_path)
    visualizer.get_representation(visualizer.environments[0])
    print('done')
