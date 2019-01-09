import random
from typing import List, Any, Union

import gym

from src.agents.agent import _Agent
from src.policy.ddqn import DoubleDeepQNetwork
from src.policy.policy import _Policy
from src.representation.learners import SimpleAutoencoder, Cerberus
from src.representation.representation import _RepresentationLearner
from src.utils.container import SARSTuple


class ParallelAgent(_Agent):
    policy: _Policy
    representation_learner: _RepresentationLearner

    def __init__(self, representation_learner: _RepresentationLearner, policy: _Policy, environment):
        self.representation_learner = representation_learner
        self.policy = policy
        self.env = environment

    # REINFORCEMENT LEARNING #

    def train_agent(self, episodes: int, max_episode_length=1000, batch_size=32, max_batch_memory_size=1024):
        print("Starting parallel training process.")

        # introduce batch memory to store observations and learn in batches
        batch_memory: List[SARSTuple] = []

        rewards: List[int] = []
        for episode in range(episodes):
            done = False
            current_state = self.env.reset()
            latent_state = self.representation_learner.encode(current_state)
            episode_reward = 0
            steps = 0
            while not done and steps < max_episode_length:
                # choose action
                action = self.policy.choose_action(latent_state)

                # step and observe
                observation, reward, done, _ = self.env.step(action)
                latent_observation = self.representation_learner.encode(observation)

                # TRAIN REPRESENTATION LEARNER using batches
                batch_memory.append(SARSTuple(current_state, action, reward, observation))
                if len(batch_memory) >= batch_size:
                    batch_tuples = batch_memory[:]
                    random.shuffle(batch_tuples)
                    batch_tuples = batch_tuples[:batch_size]

                    self.representation_learner.learn_batch_of_tuples(batch_tuples)

                    if len(batch_memory) > max_batch_memory_size:
                        batch_memory = batch_memory[1:]

                # TRAIN POLICY
                self.policy.update(latent_state, action, reward, latent_observation, done)

                # update states (both, to avoid redundant encoding)
                current_state = observation
                latent_state = latent_observation

                # trackers
                episode_reward += reward
                steps += 1

            rewards.append(episode_reward)

            if episode % (episodes // 20) == 0: print(
                f"\t|-- {round(episode/episodes * 100)}% (Avg. Rew. of {sum(rewards[-(episodes//20):])/(episodes//20)})")

        # Last update of the agent policy
        self.policy.finish_training()


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    repr_learner = Cerberus(4, 2, 3)
    policy = DoubleDeepQNetwork(3, 2)
    # size = 30
    # env = VisualObstaclePathing(size, size,
    #                             [[0, 18, 18, 21],
    #                              [21, 24, 10, 30]]
    #                             )
    # repr_learner = CerberusPixel(width=size,
    #                              height=size,
    #                              n_actions=len(env.action_space),
    #                              n_hidden=size)
    # policy = DoubleDeepQNetwork(size, len(env.action_space))
    # AGENT
    agent = ParallelAgent(repr_learner, policy, env)

    # TRAIN
    agent.train_agent(1000, max_episode_length=300)

    # TEST
    agent.test()
    agent.env.close()
