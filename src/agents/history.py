from typing import List

import gym

from src.policy.ddqn import DoubleDeepQNetwork
from src.policy.dqn import DeepQNetwork
from src.policy.policy import _Policy
from src.representation.learners import SimpleAutoencoder
from src.representation.representation import _RepresentationLearner
from src.task.task import _Task
from src.utils.container import SARSTuple


class HistoryAgent:
    history: List[SARSTuple]
    representation_learner: _RepresentationLearner
    policy: _Policy
    env: _Task

    def __init__(self, representation_learner: _RepresentationLearner, policy: _Policy, environment):
        # modules
        self.representation_learner = representation_learner
        self.policy = policy
        self.env = environment

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
            self.history.append(SARSTuple(eval(state), eval(action), eval(reward), eval(next_state)))
            tuples += 1

        print(f"Loaded {tuples} records.")

    def save_history(self, savefile: str):
        print(f"Saving to {savefile}.")
        with open(f"../../data/{savefile}", "w+") as f: pass  # clear file
        with open(f"../../data/{savefile}", "a") as f:  # write
            for sars in self.history:
                f.write(f"{sars.state.tolist()}\t{sars.action}\t{sars.reward}\t{sars.next_state.tolist()}\n")
        print("Done.")

    def gather_history(self, exploring_policy: _Policy, episodes: int, max_episode_length=1000):
        print("Gathering history...")
        rewards = []
        for episode in range(episodes):
            current_state = self.env.reset()
            done = False
            step = 0
            episode_reward = 0
            while not done and step < max_episode_length:
                action = exploring_policy.choose_action(current_state, episode)
                observation, reward, done, _ = env.step(action)
                exploring_policy.update(current_state, action, reward, observation, done)
                self.history.append(SARSTuple(current_state, action, reward, observation))
                step += 1
                episode_reward += reward
                current_state = observation
            rewards.append(episode_reward)

            if episode % (episodes // 20) == 0: print(
                f"\t|-- {round(episode/episodes * 100)}% (Avg. Rew. of {sum(rewards[-(episodes//20):])/(episodes//20)})")

        exploring_policy.finish_training()

    def pretrain(self):
        if len(self.history) == 0:
            raise RuntimeError("No history found. Add a history by using .gather_history() or .load_history()!")
        print(f"Training Representation Learner on {len(self.history)} samples ...")
        self.representation_learner.learn_many(self.history)
        self.is_pretrained = True

    # REINFORCEMENT LEARNING #

    def train_agent(self, episodes: int, max_episode_length=1000):
        print("Training Agent.")
        if not self.is_pretrained: print("[WARNING]: You are using an untrained representation learner!")
        rewards = []
        for episode in range(episodes):
            current_state = self.env.reset()
            done = False
            step = 0
            episode_reward = 0
            while not done and step < max_episode_length:
                latent_state = self.representation_learner.encode(current_state)
                action = self.policy.choose_action(latent_state, episode)

                observation, reward, done, _ = env.step(action)
                latent_observation = self.representation_learner.encode(observation)

                self.policy.update(latent_state, action, reward, latent_observation, done)
                current_state = observation

                episode_reward += reward
                step += 1

            rewards.append(episode_reward)

            if episode % (episodes // 20) == 0: print(
                f"\t|-- {round(episode/episodes * 100)}% (Avg. Rew. of {sum(rewards[-(episodes//20):])/(episodes//20)})")

        self.policy.finish_training()

    # TESTING #

    def act(self, current_state):
        current_state = self.representation_learner.encode(current_state)
        action = self.policy.choose_action_policy(current_state)
        next_state, step_reward, env_done, _ = self.env.step(action)
        return next_state, step_reward, env_done

    def test(self, max_episode_length=1000):
        done = False
        state = self.env.reset()
        step = 0
        total_reward = 0
        while not done and step < max_episode_length:
            state, reward, done = self.act(state)
            step += 1
            total_reward += reward
            env.render()

        print(f"Tested episode took {step} steps and gatherd a reward of {total_reward}.")


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    repr_learner = SimpleAutoencoder(4, 2, 3)
    policy = DoubleDeepQNetwork(3, 2, eps_decay=2000)
    pretraining_policy = DeepQNetwork(4, 2)

    agent = HistoryAgent(repr_learner, policy, env)

    load = True

    if not load:
        agent.gather_history(pretraining_policy, 10000)
        agent.save_history("cartpole-10000.data")
    else:
        agent.load_history("cartpole-10000.data")

    agent.pretrain()
    agent.train_agent(1000)

    agent.test()
    agent.env.close()