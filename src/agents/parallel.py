import random
from typing import List

import gym
import torch

import src.gym_custom_tasks

from src.agents.agent import _Agent
from src.policy.ddqn import DoubleDeepQNetwork
from src.policy.dqn import DeepQNetwork
from src.policy.policy import _Policy
from src.representation.learners import SimpleAutoencoder, CerberusPixel, JanusPixel, VariationalAutoencoder, \
    VariationalAutoencoderPixel, PassThrough, Flatten
from src.representation.representation import _RepresentationLearner
from src.representation.visual.pixelencoder import VariationalPixelEncoder
from src.utils.container import SARSTuple
from src.utils.model_handler import save_checkpoint, apply_checkpoint, get_checkpoint_dir
from src.utils.logger import Logger

class ParallelAgent(_Agent):
    policy: _Policy
    representation_learner: _RepresentationLearner

    def __init__(self, representation_learner: _RepresentationLearner, policy: _Policy, environment):
        self.representation_learner = representation_learner
        self.policy = policy
        self.env = environment
        self.one_hot_actions = torch.eye(self.env.action_space.n)
        self.logger = Logger('../../logs')

    # REINFORCEMENT LEARNING #

    def train_agent(self, episodes: int, batch_size=32, max_batch_memory_size=1024, ckpt_to_load=None,
                    save_ckpt_per=None, plot_every=None, log=False):
        start_episode = 0  # which episode to start from. This is > 0 in case of resuming training.
        if ckpt_to_load:
            start_episode = apply_checkpoint(self.policy, self.representation_learner, ckpt_to_load)

        if save_ckpt_per:  # if asked to save checkpoints
            ckpt_dir = get_checkpoint_dir(agent.get_config_name())

        print("Starting parallel training process.")
        # introduce batch memory to store observations and learn in batches
        batch_memory: List[SARSTuple] = []
        rewards = []
        for episode in range(start_episode, episodes):
            done = False

            current_state = self.reset_env()
            latent_state = self.representation_learner.encode(current_state.reshape(-1))

            episode_reward = 0

            repr_loss = 0.0

            while not done:
                # choose action
                action = self.policy.choose_action(latent_state)
                one_hot_action_vector = self.one_hot_actions[action]

                # step and observe
                observation, reward, done, _ = self.step_env(action)
                latent_observation = self.representation_learner.encode(observation)

                # TRAIN REPRESENTATION LEARNER using batches
                batch_memory.append(SARSTuple(current_state, one_hot_action_vector, reward, observation))
                if len(batch_memory) >= batch_size:
                    batch_tuples = batch_memory[:]
                    random.shuffle(batch_tuples)
                    batch_tuples = batch_tuples[:batch_size]

                    repr_loss += self.representation_learner.learn_batch_of_tuples(batch_tuples)

                    if len(batch_memory) > max_batch_memory_size:
                        batch_memory.pop(0)

                # TRAIN POLICY
                weights_before = self.representation_learner.network.encoder.weight.data.clone()
                self.policy.update(latent_state, action, reward, latent_observation, done)
                weights_after = self.representation_learner.network.encoder.weight.data.clone()

                if not torch.all(torch.eq(weights_before, weights_after)):
                    print("IT CHANGED M**********CKERS")

                # update states (both, to avoid redundant encoding)
                last_state = current_state
                current_state = observation
                latent_state = latent_observation

                # trackers
                episode_reward += reward

            # logging for tensorboard
            if log:
                info = {'loss': repr_loss, 'reward': episode_reward}
                self.logger.scalar_summary_dict(info, episode)

            rewards.append(episode_reward)

            if episode % (episodes // 100) == 0: print(
                f"\t|-- {round(episode/episodes * 100)}% (Avg. Rew. of {sum(rewards[-(episodes//100):])/(episodes//100)})")

            if save_ckpt_per and episode % save_ckpt_per == 0:  # save check point every n episodes
                save_checkpoint(self.policy.get_current_training_state(), episode, ckpt_dir, 'policy')
                save_checkpoint(self.representation_learner.current_state(), episode, ckpt_dir, 'repr')

            if plot_every is not None and episode % plot_every == 0:
                self.representation_learner.visualize_output(last_state, one_hot_action_vector, current_state)

        # Last update of the agent policy
        self.policy.finish_training()


if __name__ == "__main__":
    if torch.cuda.is_available(): torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # env = gym.make('VisualObstaclePathing-v0')  # Create VisualObstaclePathing with default values
    gym.envs.register(
        id='Evasion-v1',
        entry_point='src.gym_custom_tasks.envs:Evasion',
        kwargs={'width': 10, 'height': 10,
                'obstacle_chance': 0.01},
    )
    env = gym.make('Evasion-v1')
    # size = 30
    # gym.envs.register(
    #     id='VisualObstaclePathing-v1',
    #     entry_point='src.gym_custom_tasks.envs:ObstaclePathing',
    #     kwargs={'width': size, 'height': size,
    #             'obstacles': [[0, 18, 18, 21],
    #                           [21, 24, 10, 30]],
    #             'visual': True},
    # )
    # env = gym.make('VisualObstaclePathing-v1')

    # REPRESENTATION
    repr_learner = CerberusPixel(width=env.observation_space.shape[0],
                              height=env.observation_space.shape[1],
                              n_actions=env.action_space.n,
                              n_hidden=10)

    # repr_learner = VariationalAutoencoderPixel(width=env.observation_space.shape[0],
    #                                            height=env.observation_space.shape[1],
    #                                            n_middle=200,
    #                                            n_hidden=1)

    # repr_learner = Flatten()

    # POLICY
    policy = DoubleDeepQNetwork(10, env.action_space.n, eps_decay=2000, representation_network=repr_learner.network)

    # AGENT
    agent = ParallelAgent(repr_learner, policy, env)

    # TRAIN
    agent.train_agent(episodes=10000, plot_every=None, log=False)

    # TEST
    # Gifs will only be produced when render is off
    agent.test(num_testruns=5, render=False)
    agent.env.close()
