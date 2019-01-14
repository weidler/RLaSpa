import random
import time
from collections import deque

import gym
import torch
from gym import Env

from src.agents.agent import _Agent
from src.policy.ddqn import DoubleDeepQNetwork
from src.policy.policy import _Policy
from src.representation.learners import CerberusPixel
from src.representation.representation import _RepresentationLearner
from src.utils.container import SARSTuple
from src.utils.model_handler import save_checkpoint, apply_checkpoint


class ParallelAgent(_Agent):

    def __init__(self, representation_learner: _RepresentationLearner, policy: _Policy, environment: Env,
                 representation_memory_size: int = 1024) -> None:
        """
        Initializes a parallel agent with the required elements. The parallel agent trains at the same time
        the representation and the policy learners.

        :param representation_learner: module that converts an environment state into a latent representation.
        :param policy: module that will make the decisions about how the agent is going to act.
        :param environment: environment in which the agent acts.
        :param representation_memory_size: number of SARS tuples that the representation learning memory can save.
        """
        super().__init__(representation_learner=representation_learner, policy=policy, environment=environment)
        self.one_hot_actions = torch.eye(self.env.action_space.n)
        self.representation_memory = deque(maxlen=representation_memory_size)

    def train_agent(self, episodes: int, batch_size: int = 32, ckpt_to_load: str = None,
                    episodes_per_saving: int = None, plot_every: int = None, log: bool = False) -> None:
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
        start_episode = 0  # which episode to start from. This is > 0 in case of resuming training.
        start_time = time.time()
        if not (ckpt_to_load is None):
            self.start_episode = apply_checkpoint(ckpt_to_load, policy=self.policy, repr=self.representation_learner)
        if not (episodes_per_saving is None):  # if asked to save checkpoints
            ckpt_dir = self.path_manager.get_ckpt_idr(self.get_config_name())
        else:
            ckpt_dir = None
        print("Starting parallel training process.")
        # introduce batch memory to store observations and learn in batches

        rewards = []
        all_repr_loss = []
        all_policy_loss = []
        for episode in range(self.start_episode, episodes):
            done = False

            current_state = self.reset_env()
            latent_state = self.representation_learner.encode(current_state.reshape(-1))

            episode_reward = 0

            repr_loss = 0
            policy_loss = 0

            while not done:
                # choose action
                action = self.policy.choose_action(latent_state)
                one_hot_action_vector = self.one_hot_actions[action]
                # step and observe
                observation, reward, done, _ = self.step_env(action)
                latent_observation = self.representation_learner.encode(observation)
                # TRAIN REPRESENTATION LEARNER using batches
                sars = SARSTuple(current_state, one_hot_action_vector, reward, observation)
                self.representation_memory.append(sars)
                if len(self.representation_memory) >= batch_size:
                    batch_tuples = random.sample(self.representation_memory, batch_size)
                    repr_loss += self.representation_learner.learn_batch_of_tuples(batch_tuples)
                # TRAIN POLICY
                policy_loss += self.policy.update(latent_state, action, reward, latent_observation, done)
                # update states (both, to avoid redundant encoding)
                last_state = current_state
                current_state = observation
                latent_state = latent_observation
                # trackers
                episode_reward += reward
            # logging for tensorboard
            if log:
                info = {'loss': repr_loss, 'policy_loss': policy_loss, 'reward': episode_reward}
                self.logger.scalar_summary_dict(info, episode)
            rewards.append(episode_reward)
            all_repr_loss.append(repr_loss)
            all_policy_loss.append(policy_loss)

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
                # save check point every n episodes
                save_checkpoint(self.policy.get_current_training_state(), episode, ckpt_dir, 'policy')
                save_checkpoint(self.representation_learner.current_state(), episode, ckpt_dir, 'repr')

            if not (plot_every is None) and episode % plot_every == 0:
                self.representation_learner.visualize_output(last_state, one_hot_action_vector, current_state)

        # Last update of the agent policy
        self.policy.finish_training()


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # env = gym.make('VisualObstaclePathing-v0')  # Create VisualObstaclePathing with default values
    size = 30
    env = gym.make('Evasion-v0')
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
                                 n_hidden=size)

    # repr_learner = VariationalAutoencoderPixel(width=env.observation_space.shape[0],
    #                                            height=env.observation_space.shape[1],
    #                                            n_middle=200,
    #                                            n_hidden=1)

    # repr_learner = Flatten()

    # POLICY
    policy = DoubleDeepQNetwork(size, env.action_space.n, eps_decay=2000)

    # AGENT
    agent = ParallelAgent(repr_learner, policy, env)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    repr_learner.network.to(device)  # if using passthrough or Flatten comment this
    policy.current_model.to(device)
    policy.target_model.to(device)

    # TRAIN
    start_time = time.time()
    # LOAD
    # agent.load('../../ckpt/ParallelAgent_Evasion_CerberusPixel_DoubleDeepQNetwork/2019-01-13_20-55-49')
    total_episodes = 10000
    agent.train_agent(episodes=total_episodes, log=True, episodes_per_saving=500)
    print(f'Total training took {(time.time()-start_time)/60:.2f} min')
    # SAVE
    # agent.save(total_episodes-1)
    # TEST
    # Gifs will only be produced when render is off
    agent.test(runs_number=5, render=False)
    agent.env.close()
