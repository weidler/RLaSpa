import cv2
import random
from typing import List, Tuple

import gym
import numpy as np
import torch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn import manifold

import src.gym_custom_tasks
from torch import Tensor

from src.agents.agent import reset_env, step_env
from src.policy.dqn import DoubleDeepQNetwork
import matplotlib.pyplot as plt
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
                                 gym.make("EvasionWalls-v0")]
        elif env_name == 'all':
            self.environments = [gym.make("Tunnel-v0"),
                                 gym.make("Evasion-v0"),
                                 gym.make("EvasionWalls-v0"),
                                 gym.make("Race-v0")]
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
                           steps_per_snapshot: int) -> Tuple[np.ndarray, np.ndarray]:
        state = reset_env(environment)
        steps = 0
        snapshots_taken = 0
        states = np.empty((number_of_snapshots, self.height, self.width))
        latent_representations = np.empty((number_of_snapshots, 32))
        while snapshots_taken < number_of_snapshots:
            action = self.policy.choose_action_policy(state)
            next_state, _, done, _ = step_env(action=action, env=environment)
            steps += 1
            if steps % steps_per_snapshot == 0:
                states[snapshots_taken] = state.data.cpu().numpy()
                latent_representation = self.representation_module.encode(state)
                latent_representations[snapshots_taken] = latent_representation.data.cpu().numpy()
                one_hot_action_vector = self.one_hot_actions[action]
                self.representation_module.visualize_output(state, one_hot_action_vector, next_state)
                snapshots_taken += 1
            if done:
                state = reset_env(environment)
            else:
                state = next_state
        return states, latent_representations

    # Scatter with images instead of points
    def imscatter(self, x, y, ax, imageData, zoom, latent_repr: bool = False):
        images = []
        for i in range(len(x)):
            x0, y0 = x[i], y[i]
            # Convert to image
            img = (1- imageData[i]) * 255.
            if latent_repr:
                size = 32
            else:
                size = [self.width, self.height]
            img_in = img.astype(np.uint8).reshape(size)
            boarder = 1
            img = np.zeros((img_in.shape[0] + boarder*2, img_in.shape[1] + boarder*2))
            img[boarder: boarder+img_in.shape[0], boarder: boarder+img_in.shape[1]] = img_in
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            # Note: OpenCV uses BGR and plt uses RGB
            image = OffsetImage(img, zoom=zoom)
            ab = AnnotationBbox(image, (x0, y0), xycoords='data', frameon=False)
            images.append(ax.add_artist(ab))

        ax.update_datalim(np.column_stack([x, y]))
        ax.autoscale()

    def compute_tsne_states(self, states, latent_repr: bool = False, display: bool = True):
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        if not latent_repr:
            states = states.reshape([-1, self.width * self.height * 1])
        states_tsne = tsne.fit_transform(states)

        # Plot images according to t-sne embedding
        if display:
            print("Plotting t-SNE visualization...")
            fig, ax = plt.subplots()
            self.imscatter(states_tsne[:, 0], states_tsne[:, 1], imageData=states, ax=ax, zoom=0.6,
                           latent_repr=latent_repr)
            plt.show()
        else:
            return states_tsne

    def represent_latent_space(self, states: np.ndarray, latent_rep: np.ndarray, display: bool = True):
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        latent_tsne = tsne.fit_transform(latent_rep)
        # Plot images according to t-sne embedding
        if display:
            print("Plotting t-SNE visualization...")
            fig, ax = plt.subplots()
            self.imscatter(latent_tsne[:, 0], latent_tsne[:, 1], imageData=states, ax=ax, zoom=0.6)
            plt.show()
        else:
            return latent_tsne


if __name__ == '__main__':
    env_name = 'all'
    repr_learner_name = 'cerberus'
    # savefiles names
    tunel_ckpt = 'ParallelAgent_Tunnel-v0'
    scrollers_ckpt = 'ParallelAgent_Tunnel-v0_Evasion-v0_EvasionWalls-v0'
    cerberus_ckpt = '_CerberusPixel_DoubleDeepQNetwork'
    janus_ckpt = '_JanusPixel_DoubleDeepQNetwork'
    cvae_ckpt = '_CVAEPixel_DoubleDeepQNetwork'
    # Final path
    ckpt_path = '/home/adrigrillo/Documents/RLaSpa/ckpt/' + scrollers_ckpt + cerberus_ckpt
    # Load environments and representation learner
    visualizer = RepresentationViewer(env_name=env_name, repr_learner_name=repr_learner_name,
                                      ckpt_path=ckpt_path, load_policy=False)
    # Graph configuration
    number_of_snapshots = 20
    states = np.empty((number_of_snapshots * len(visualizer.environments), visualizer.height, visualizer.width))
    latent_repr = np.empty((number_of_snapshots * len(visualizer.environments), 32))

    for i in range(len(visualizer.environments)):
        new_states, new_latent_repr = visualizer.get_representation(environment=visualizer.environments[i],
                                                                    number_of_snapshots=number_of_snapshots,
                                                                    steps_per_snapshot=5)
        states[i*number_of_snapshots:(i+1)*number_of_snapshots] = new_states
        latent_repr[i * number_of_snapshots:(i + 1) * number_of_snapshots] = new_latent_repr
    # visualize
    visualizer.compute_tsne_states(states)
    visualizer.represent_latent_space(states, latent_repr)
