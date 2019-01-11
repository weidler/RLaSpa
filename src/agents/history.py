import random
from typing import List

import gym
import torch

from src.agents.agent import _Agent
from src.gym_custom_tasks.envs import ObstaclePathing
from src.policy.ddqn import DoubleDeepQNetwork
from src.policy.dqn import DeepQNetwork
from src.policy.policy import _Policy
from src.representation.learners import SimpleAutoencoder, VariationalAutoencoder, JanusPixel, Flatten
from src.representation.representation import _RepresentationLearner
from src.utils.container import SARSTuple
from src.utils.model_handler import save_checkpoint, apply_checkpoint, get_checkpoint_dir


class HistoryAgent(_Agent):
    history: List[SARSTuple]
    representation_learner: _RepresentationLearner
    policy: _Policy
    env: gym.Env

    def __init__(self, representation_learner: _RepresentationLearner, policy: _Policy, environment: gym.Env):
        # modules
        self.representation_learner = representation_learner
        self.policy = policy
        self.env = environment
        self.one_hot_actions = torch.eye(self.env.action_space.n)

        # quick check ups
        # TODO check if policy input size matches representation encoding size

        self.history = []
        self.is_pretrained = False

    # REPRESENTATION LEARNING #

    def load_history(self, savefile: str):
        print(f"Loading from {savefile}.")
        with open(f"../../data/{savefile}", "r") as f:  # write
            lines = f.readlines()

        tuples = 0
        for line in lines:
            state, action, reward, next_state = line.split("\t")
            self.history.append(SARSTuple(torch.Tensor(eval(state)).float(), torch.Tensor(eval(action)).float(), eval(reward), torch.Tensor(eval(next_state)).float()))
            tuples += 1

        print(f"Loaded {tuples} records.")

    def save_history(self, savefile: str):
        print(f"Saving to {savefile}.")
        with open(f"../../data/{savefile}", "w+") as f: pass  # clear file
        with open(f"../../data/{savefile}", "a") as f:  # write
            for sars in self.history:
                f.write(f"{sars.state.tolist()}\t{sars.action.tolist()}\t{sars.reward}\t{sars.next_state.tolist()}\n")
        print("Done.")

    def gather_history(self, exploring_policy: _Policy, episodes: int, max_episode_length=1000):
        print("Gathering history...")
        # check if it is a visual task that needs flattening
        is_visual = False
        flattener = Flatten()
        if len(self.env.observation_space.shape) == 2:
            is_visual = True


        rewards = []
        for episode in range(episodes):
            current_state = self.reset_env()
            done = False
            step = 0
            episode_reward = 0
            while not done and step < max_episode_length:
                action = exploring_policy.choose_action(flattener.encode(current_state))
                one_hot_action_vector = self.one_hot_actions[action]
                observation, reward, done, _ = self.step_env(action)
                if is_visual: exploring_policy.update(flattener.encode(current_state), action, reward, flattener.encode(observation), done)
                else: exploring_policy.update(current_state, action, reward, observation, done)
                self.history.append(SARSTuple(current_state, one_hot_action_vector, reward, observation))
                step += 1
                episode_reward += reward
                current_state = observation
            rewards.append(episode_reward)

            if episode % (episodes // 20) == 0: print(
                f"\t|-- {round(episode/episodes * 100)}% (Avg. Rew. of {sum(rewards[-(episodes//20):])/(episodes//20)})")

        exploring_policy.finish_training()

    def pretrain(self, epochs: int, batch_size=32):
        if len(self.history) == 0:
            raise RuntimeError("No history found. Add a history by using .gather_history() or .load_history()!")

        print(f"Training Representation Learner on {len(self.history)} samples ...")

        print("\t|-- Shuffling")
        random.shuffle(self.history)

        print("\t|-- Training")
        n_batches = len(self.history) // batch_size
        for epoch in range(epochs):
            print(f"\t\tEpoch {epoch + 1}")
            for i in range(n_batches):
                batch_tuples = self.history[i * batch_size:(i + 1) * batch_size]

                self.representation_learner.learn_batch_of_tuples(batch_tuples)

                if i % (n_batches // 3) == 0: print(
                    f"\t\t|-- {round(i/n_batches * 100)}%")

        self.is_pretrained = True

    # REINFORCEMENT LEARNING #

    def train_agent(self, episodes: int, max_episode_length=1000, ckpt_to_load=None, save_ckpt_per=None):
        start_episode = 0  # which episode to start from. This is > 0 in case of resuming training.
        if ckpt_to_load:
            start_episode = apply_checkpoint(self.policy, self.representation_learner, ckpt_to_load)

        if save_ckpt_per:  # if asked to save checkpoints
            ckpt_dir = get_checkpoint_dir(agent.get_config_name())

        print("Training Agent.")
        if not self.is_pretrained: print("[WARNING]: You are using an untrained representation learner!")
        rewards = []
        for episode in range(start_episode, episodes):
            current_state = self.reset_env()
            done = False
            step = 0
            episode_reward = 0
            while not done and step < max_episode_length:
                latent_state = self.representation_learner.encode(current_state)
                action = self.policy.choose_action(latent_state)

                observation, reward, done, _ = self.step_env(action)
                latent_observation = self.representation_learner.encode(observation)

                self.policy.update(latent_state, action, reward, latent_observation, done)

                current_state = observation

                episode_reward += reward
                step += 1

            rewards.append(episode_reward)

            if episode % (episodes // 20) == 0: print(
                f"\t|-- {round(episode/episodes * 100)}% (Avg. Rew. of {sum(rewards[-(episodes//20):])/(episodes//20)})")

            if save_ckpt_per and episode % save_ckpt_per == 0:  # save check point every n episodes
                save_checkpoint(self.policy.get_current_training_state(), episode, ckpt_dir, 'policy')
                save_checkpoint(self.representation_learner.current_state(), episode, ckpt_dir, 'repr')

        self.policy.finish_training()


if __name__ == "__main__":
    # env = gym.make("CartPole-v0")
    # repr_learner = SimpleAutoencoder(4, 2, 3)
    # policy = DoubleDeepQNetwork(3, 2, eps_decay=2000)
    # pretraining_policy = DeepQNetwork(4, 2)
    #
    # agent = HistoryAgent(repr_learner, policy, env)
    #
    # load = True
    #
    # if not load:
    #     agent.gather_history(pretraining_policy, 10000)
    #     agent.save_history("cartpole-10000.data")
    # else:
    #     agent.load_history("cartpole-10000.data")
    #
    # agent.pretrain(5)
    # agent.train_agent(1000)
    #
    # agent.test()
    # agent.env.close()


    size=30

    env = ObstaclePathing(30, 30,
                          [[0, 18, 18, 21],
                           [21, 24, 10, 30]],
                          True
                          )

    repr_learner = JanusPixel(width=size,
                              height=size,
                              n_actions=env.action_space.n,
                              n_hidden=size)
    policy = DoubleDeepQNetwork(30, 2, eps_decay=2000)

    pretraining_policy = DeepQNetwork(900, 2)

    agent = HistoryAgent(repr_learner, policy, env)

    load = True
    if not load:
        agent.gather_history(pretraining_policy, 20)
        agent.save_history("cartpole-10.data")
    else:
        agent.load_history("cartpole-10.data")

    agent.pretrain(5)
    agent.train_agent(1000)

    for i in range(5): agent.test()
    agent.env.close()
