import random
from typing import List

import gym
import torch
from gym import Env

from src.agents.agent import _Agent, reset_env, step_env
from src.gym_custom_tasks.envs import ObstaclePathing
from src.policy.dqn import DeepQNetwork
from src.policy.dqn_prioritized import PrioritizedDoubleDeepQNetwork
from src.policy.policy import _Policy
from src.representation.learners import JanusPixel, Flatten
from src.representation.representation import _RepresentationLearner
from src.utils.container import SARSTuple
from src.utils.model_handler import save_checkpoint, apply_checkpoint
from src.utils.schedules import LinearSchedule, ExponentialSchedule


class HistoryAgent(_Agent):
    history: List[SARSTuple]
    representation_learner: _RepresentationLearner
    policy: _Policy
    environments: List[gym.Env]

    def __init__(self, representation_learner: _RepresentationLearner, policy: _Policy, environment: Env):
        """
        Initializes a history agent with the required elements. The history will explore the environment and save
        the execution history. Then, it will use the history to train the representation learner and, finally, will
        use the trained representation learner to train the policy learner with its latent space.

        :param representation_learner: module that converts an environment state into a latent representation.
        :param policy: module that will make the decisions about how the agent is going to act.
        :param environment: environment in which the agent acts.
        """
        # modules
        super().__init__(representation_learner, policy, environment)
        self.one_hot_actions = torch.eye(self.environments[0].action_space.n)

        # quick check ups
        # TODO check if policy input size matches representation encoding size

        self.history = []
        self.is_pretrained = False

    def load_history(self, file_name: str) -> None:
        """
        Method that read the history file and load the state, action, reward, next_state tuples in to the execution.

        :param file_name: name of the file where the history is saved.
        """
        print(f"Loading from {file_name}.")
        with open(f"../../data/{file_name}", "r") as f:
            lines = f.readlines()
        tuples = 0
        for line in lines:
            state, action, reward, next_state = line.split("\t")
            sars = SARSTuple(torch.Tensor(eval(state)).float(), torch.Tensor(eval(action)).float(), eval(reward),
                             torch.Tensor(eval(next_state)).float())
            self.history.append(sars)
            tuples += 1
        print(f"Loaded {tuples} records.")

    def save_history(self, file_name: str) -> None:
        """
        Method that copy the state, action, reward, next_state tuples for the execution and saves them in a file.

        :param file_name: name of the file where the history is saved.
        """
        print(f"Saving to {file_name}.")
        with open(f"../../data/{file_name}", "w+") as f:
            pass  # clear file
        with open(f"../../data/{file_name}", "a") as f:  # write
            for sars in self.history:
                f.write(f"{sars.state.tolist()}\t{sars.action.tolist()}\t{sars.reward}\t{sars.next_state.tolist()}\n")
        print("Done.")

    def gather_history(self, exploring_policy: _Policy, episodes: int, max_episode_length: int = 1000,
                       log: bool = False) -> None:
        """
        Method that executes the environment to generate a history of state, action, reward, next_state tuples using
        only the policy learner specified.

        :param exploring_policy: policy used during the gathering execution.
        :param episodes: number of episodes to gather.
        :param max_episode_length: max number of steps for the episodes.
        :param log: logging flag.
        """
        print("Gathering history...")
        # check if it is a visual task that needs flattening
        is_visual = False
        flattener = Flatten()
        if len(self.environments.observation_space.shape) == 2:
            is_visual = True

        rewards = []
        for episode in range(episodes):
            # choose environment
            env = random.choice(self.environments)
            current_state = reset_env(env)

            # trackers
            step = 0
            episode_reward = 0

            # start episode
            done = False
            exp_policy_loss = 0
            while not done and step < max_episode_length:
                encoded_current_state = flattener.encode(current_state)
                action = exploring_policy.choose_action(encoded_current_state)
                one_hot_action_vector = self.one_hot_actions[action]

                next_state, reward, done, _ = step_env(action, env)
                if is_visual:
                    encoded_next_state = flattener.encode(next_state)
                    exp_policy_loss += exploring_policy.update(encoded_current_state, action, reward,
                                                               encoded_next_state, done)
                else:
                    exp_policy_loss += exploring_policy.update(current_state, action, reward, next_state, done)
                sars = SARSTuple(current_state, one_hot_action_vector, reward, next_state)
                self.history.append(sars)

                step += 1
                episode_reward += reward
                current_state = next_state
            rewards.append(episode_reward)

            if episode % (episodes // 20) == 0:
                print(f"\t|-- {round(episode / episodes * 100)}% " +
                      f"(Avg. Rew. of {sum(rewards[-(episodes // 20):]) / (episodes // 20)})")

            if log:
                info = {'explore_policy_loss': exp_policy_loss, 'explore_reward': episode_reward}
                self.logger.scalar_summary_dict(info, episode)

        exploring_policy.finish_training()

    def pretrain(self, epochs: int, batch_size: int = 32, log: bool = False) -> None:
        """
        Method that train the representation learner with the previously gathered data.

        :param epochs: number of times the data will be used to train.
        :param batch_size: number of samples used to train in each training iteration.
        :param log: logging flag.
        """
        if len(self.history) == 0:
            raise RuntimeError("No history found. Add a history by using .gather_history() or .load_history()!")

        print(f"Training Representation Learner on {len(self.history)} samples ...")

        print("\t|-- Shuffling")
        random.shuffle(self.history)

        print("\t|-- Training")
        number_of_batches = len(self.history) // batch_size
        for epoch in range(epochs):
            print(f"\t\tEpoch {epoch + 1}")
            pretrain_loss = 0
            for batch_number in range(number_of_batches):
                batch_tuples = self.history[batch_number * batch_size:(batch_number + 1) * batch_size]
                pretrain_loss += self.representation_learner.learn_batch_of_tuples(batch_tuples)
                if batch_number % (number_of_batches // 3) == 0: print(
                    f"\t\t|-- {round(batch_number / number_of_batches * 100)}%")
            if log:
                self.logger.scalar_summary('pretrain_loss', pretrain_loss, epoch)
        self.is_pretrained = True

    def train_agent(self, episodes: int, max_episode_length: int = 1000, ckpt_to_load: str = None,
                    episodes_per_saving: int = None, log: bool = False) -> None:
        """
        Method that trains the agent policy learner using the pretrained representation learner.

        :param episodes: number of training episodes.
        :param max_episode_length: max number of steps for the episodes.
        :param ckpt_to_load: name of the checkpoint to load a pretrained policy learner.
        :param episodes_per_saving: number of episodes between saving checkpoint.
        :param log: logging flag.
        """
        if not (ckpt_to_load is None):
            self.load(ckpt_dir=ckpt_to_load)
        if not (episodes_per_saving is None):  # if asked to save checkpoints
            ckpt_dir = self.path_manager.get_ckpt_dir(agent.get_config_name())
        else:
            ckpt_dir = None
        print("Training Agent.")
        if not self.is_pretrained:
            print("[WARNING]: You are using an untrained representation learner!")
        rewards = []

        for episode in range(self.start_episode, episodes):
            env = random.choice(self.environments)

            current_state = reset_env(env)
            done = False
            step = 0
            episode_reward = 0
            policy_loss = 0
            while not done and step < max_episode_length:
                latent_state = self.representation_learner.encode(current_state)
                action = self.policy.choose_action(latent_state)

                observation, reward, done, _ = step_env(action, env)
                latent_observation = self.representation_learner.encode(observation)

                policy_loss += self.policy.update(latent_state, action, reward, latent_observation, done)

                current_state = observation

                episode_reward += reward
                step += 1
            rewards.append(episode_reward)

            if episode % (episodes // 20) == 0:
                print(f"\t|-- {round(episode / episodes * 100)}% " +
                      f"(Avg. Rew. of {sum(rewards[-(episodes // 20):]) / (episodes // 20)})")

            if episodes_per_saving and episode % episodes_per_saving == 0 and episode != 0:
                # save check point every n episodes
                self.save(episode=episode)

            if log:
                info = {'policy_loss': policy_loss, 'reward': episode_reward}
                self.logger.scalar_summary_dict(info, episode)

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

    size = 30

    env = ObstaclePathing(30, 30,
                          [[0, 18, 18, 21],
                           [21, 24, 10, 30]],
                          True
                          )

    repr_learner = JanusPixel(width=size,
                              height=size,
                              n_actions=env.action_space.n,
                              n_hidden=size)
    memory_delay = 5000
    init_eps = 1.0
    memory_eps = 0.8
    min_eps = 0.01
    eps_decay = 10000
    linear = LinearSchedule(schedule_timesteps=memory_delay, initial_p=init_eps, final_p=memory_eps)
    exponential = ExponentialSchedule(initial_p=memory_eps, min_p=min_eps, decay=eps_decay)
    policy = PrioritizedDoubleDeepQNetwork(20, env.action_space.n, eps_calculator=linear,
                                           memory_eps_calculator=exponential, memory_delay=memory_delay)

    pretraining_policy = DeepQNetwork(900, 2, eps_calculator=linear, memory_eps_calculator=exponential,
                                      memory_delay=memory_delay)

    agent = HistoryAgent(repr_learner, policy, env)

    load = True
    if not load:
        agent.gather_history(pretraining_policy, 20)
        agent.save_history("cartpole-10.data")
    else:
        agent.load_history("cartpole-10.data")

    agent.pretrain(5)
    # SAVE
    # agent.save(episode=0, save_policy_learner=False)
    # LOAD
    # agent.load(ckpt_dir='../../ckpt/HistoryAgent_ObstaclePathing_JanusPixel_DoubleDeepQNetwork/2019-01-13_18-17-16',
    #            load_policy_learner=False)

    agent.train_agent(1000)

    for i in range(5): agent.test()
    agent.environments.close()
