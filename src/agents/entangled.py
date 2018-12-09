import copy

import gym
import numpy

from src.policy.tablebased import QTable
from src.representation.autoencoder import Autoencoder
from src.task.pathing import SimplePathing
from src.utils.exploration import boltzmann_explore


class EntangledAgent(object):

    def __init__(self, representation_learner, policy, environment):
        self.representation_learner = representation_learner
        self.policy = policy
        self.env = environment

    def train_representation_learner(self):
        pass

    def train_policy(self, current_state):
        action = self.policy.choose_action(current_state)

        # observe
        observation, reward, done = self.env.step(action)

        # update policy
        self.policy.update(current_state, action, reward, observation)

        return observation, done

    def act(self, state):
        action = self.policy.choose_action(state)
        observation, reward, done = self.env.step(action)

        return observation, done

if __name__ == "__main__":
    env = SimplePathing(10, 10)
    repr_learner = Autoencoder()
    policy = QTable([env.height, env.width], len(env.action_space))

    # AGENT
    agent = EntangledAgent(repr_learner, policy, env)

    # TRAIN
    epochs = 10000
    for epoch in range(epochs):
        done = False
        state = env.reset()
        while not done:
            state, done = agent.train_policy(state)

        print(epoch)

    # TEST
    max_steps = 1000
    done = False
    state = env.reset()
    while not done and max_steps > 0:
        state, done = agent.act(state)
        max_steps -= 1

    env.show_breadcrumbs = True
    print(env.target_coords)
    print(env)

