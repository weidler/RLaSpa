import random
import time
from collections import deque
from typing import List

import gym
import torch
from gym import Env

from src.agents.agent import _Agent, reset_env, step_env
from src.policy.ddqn import DoubleDeepQNetwork
from src.policy.policy import _Policy
from src.representation.learners import CerberusPixel
from src.representation.representation import _RepresentationLearner
from src.utils.container import SARSTuple
from src.utils.model_handler import save_checkpoint, apply_checkpoint


class ParallelAgent(_Agent):
    policy: _Policy
    representation_learner: _RepresentationLearner
    environments: List[Env]

    def __init__(self, representation_learner: _RepresentationLearner, policy: _Policy, environments: List[Env],
                 representation_memory_size: int = 1024):
        """ Initialize a parallel agent with the required elements. The parallel agent trains at the same time
        the representation and the policy learners.

        :param representation_learner: module that converts an environment state into a latent representation.
        :param policy: module that will make the decisions about how the agent is going to act.
        :param environment: environment in which the agent acts.
        :param representation_memory_size: number of SARS tuples that the representation learning memory can save.
        """

        super(ParallelAgent, self).__init__(representation_learner, policy, environments)

        self.one_hot_actions = torch.eye(self.environments[0].action_space.n)
        self.representation_memory = deque(maxlen=representation_memory_size)

        print(f"Created Agent using {self.representation_learner.__class__.__name__} and {self.policy.__class__.__name__}.")

    # REINFORCEMENT LEARNING #

    def train_agent(self, episodes: int, batch_size: int = 32, ckpt_to_load: str = None,
                    episodes_per_saving: int = None, plot_every: int = None, numb_intermediate_tests: int = 0,
                    experience_warmup_length=0, log: bool = False) -> None:
        """
        Method that trains the agent policy learner using the pretrained representation learner.

        :param episodes: number of training episodes.
        :param batch_size: number of samples used to train in each training iteration.
        :param ckpt_to_load: name of the checkpoint to load a pretrained policy learner.
        :param episodes_per_saving: number of episodes between saving checkpoint.
        :param plot_every: number of steps that will happen between the plotting of the space representation.
        :param log: logging flag.
        """

        episodes_per_report = episodes // 100
        start_time = time.time()
        if not (ckpt_to_load is None):
            self.start_episode = apply_checkpoint(ckpt_to_load, policy=self.policy, repr=self.representation_learner)
        if not (episodes_per_saving is None):  # if asked to save checkpoints
            ckpt_dir = self.path_manager.get_ckpt_idr(self.get_config_name())
        else:
            ckpt_dir = None

        print("Starting parallel training process.")
        rewards = []
        all_repr_loss = []
        all_policy_loss = []

        # store experiences in a stack to allow random shuffling between the episodes (and therefore between different
        # tasks.
        experience_stack = []
        for episode in range(self.start_episode, episodes):
            # choose environment
            env = random.choice(self.environments)

            # initialize episode
            current_state = reset_env(env)
            latent_state = self.representation_learner.encode(current_state.reshape(-1))

            # trackers
            episode_reward = 0
            repr_loss = 0
            policy_loss = 0

            # begin episode
            done = False
            while not done:
                # choose action
                action = self.policy.choose_action(latent_state)
                one_hot_action_vector = self.one_hot_actions[action]

                # step and observe
                observation, reward, done, _ = step_env(action, env)

                # store sars tuple for batch learning
                sars_tuple = SARSTuple(current_state, one_hot_action_vector, reward, observation)
                self.representation_memory.append(sars_tuple)

                # train representation module
                if len(self.representation_memory) >= batch_size:
                    batch_tuples = random.sample(self.representation_memory, batch_size)
                    repr_loss += self.representation_learner.learn_batch_of_tuples(batch_tuples)

                # store experience for multitask learning
                experience = (current_state, action, reward, observation, done)
                experience_stack.append(experience)

                # train policy
                if episode >= experience_warmup_length:
                    # transformations into the latent space need to be done here such that the update is based on the
                    # current state of the encoder!
                    exp_state, exp_action, exp_reward, exp_next_state, exp_done = experience_stack.pop(
                        random.randint(0, len(experience_stack) - 1))
                    exp_latent_state = self.representation_learner.encode(exp_state)
                    exp_latent_observation = self.representation_learner.encode(exp_next_state)
                    policy_loss += self.policy.update(exp_latent_state, exp_action, exp_reward, exp_latent_observation,
                                                      exp_done)

                # update states (both, to avoid redundant encoding)
                last_state = current_state
                current_state = observation

                # trackers
                episode_reward += reward

            rewards.append(episode_reward)
            all_repr_loss.append(repr_loss)
            all_policy_loss.append(policy_loss)

            # logging for tensorboard
            if log:
                info = {'loss': repr_loss, 'policy_loss': policy_loss, 'reward': episode_reward}
                self.logger.scalar_summary_dict(info, episode)

            # progress report
            if episode % (episodes_per_report) == 0:
                last_episodes_rewards = rewards[-(episodes_per_report):]
                print(f"\t|-- {round(episode / episodes * 100):3d}%; " \
                      + f"r-avg: {(sum(last_episodes_rewards) / (episodes_per_report)):8.2f}; r-peak: {max(last_episodes_rewards):4d};"
                        f" r-slack: {min(last_episodes_rewards):4d}; r-common: {max(set(last_episodes_rewards), key=last_episodes_rewards.count):4d}; " \
                      + f"Avg. repr_loss: {sum(all_repr_loss[-(episodes_per_report):]) / (episodes_per_report):10.4f}; " \
                      + f"Avg. policy_loss: {sum(all_policy_loss[-(episodes_per_report):]) / (episodes_per_report):15.4f}; " \
                      + f"Time elapsed: {(time.time()-start_time)/60:6.2f} min; " \
                      + f"Eps: {self.policy.memory_epsilon_calculator.value(self.policy.total_steps_done - self.policy.memory_delay):.5f}")

            if not (episodes_per_saving is None) and episode % episodes_per_saving == 0:
                save_checkpoint(self.policy.get_current_training_state(), episode, ckpt_dir, 'policy')
                save_checkpoint(self.representation_learner.current_state(), episode, ckpt_dir, 'repr')

            # plotting the representation heads
            if not (plot_every is None) and episode % plot_every == 0:
                self.representation_learner.visualize_output(last_state, one_hot_action_vector, current_state)

        # Last update of the agent policy
        self.policy.finish_training()


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    environments = [
        gym.make("VisualObstaclePathing-v0"),
        gym.make("VisualObstaclePathing-v1"),
        gym.make("VisualObstaclePathing-v2"),
        gym.make("VisualObstaclePathing-v3")
    ]

    # REPRESENTATION

    repr_learner = CerberusPixel(width=environments[0].observation_space.shape[0],
                                 height=environments[0].observation_space.shape[1],
                                 n_actions=environments[0].action_space.n,
                                 n_hidden=20)

    # POLICY
    policy = DoubleDeepQNetwork(20, environments[0].action_space.n, eps_decay=5000)

    # AGENT
    agent = ParallelAgent(repr_learner, policy, environments)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    repr_learner.network.to(device)  # if using passthrough or Flatten comment this
    policy.current_model.to(device)
    policy.target_model.to(device)

    # TRAIN
    start_time = time.time()

    agent.train_agent(episodes=100, batch_size=32, plot_every=10, log=False)
    print(f'Total training took {(time.time()-start_time)/60:.2f} min')

    # TEST
    # Gifs will only be produced when render is off
    agent.test(environments[0], numb_runs=5, render=False)
    for env in environments:
        env.close()
