import gym
import os

from src.agents.agent import _Agent
from src.policy.ddqn import DoubleDeepQNetwork
from src.policy.policy import _Policy
from src.representation.learners import SimpleAutoencoder, CerberusPixel
from src.representation.representation import _RepresentationLearner
from src.utils.model_handler import save_checkpoint, load_checkpoint, get_checkpoint_dir


class ParallelAgent(_Agent):
    policy: _Policy
    representation_learner: _RepresentationLearner

    def __init__(self, representation_learner: _RepresentationLearner, policy: _Policy, environment):
        self.representation_learner = representation_learner
        self.policy = policy
        self.env = environment

    # REINFORCEMENT LEARNING #


    def train_agent(self, episodes: int, ckpt_to_load=None, save_ckpt_per=None):
        start_episode = 0  # which episode to start from. This is > 0 in case of resuming training.
        if ckpt_to_load:
            start_episode = load_checkpoint(policy, ckpt_to_load)

        if save_ckpt_per:  # if asked to save checkpoints
            ckpt_dir = get_checkpoint_dir(agent.get_config_name())

        print("Starting parallel training process.")
        rewards = []
        for episode in range(start_episode, episodes):
            done = False
            current_state = self.env.reset()
            latent_state = self.representation_learner.encode(current_state)
            episode_reward = 0
            while not done:
                # choose action
                action = self.policy.choose_action(latent_state)

                # step and observe
                observation, reward, done, _ = self.env.step(action)
                latent_observation = self.representation_learner.encode(observation)

                # train the REPRESENTATION learner
                self.representation_learner.learn(state=[current_state], next_state=[observation], reward=[reward],
                                                  action=[action])

                # train the POLICY
                self.policy.update(latent_state, action, reward, latent_observation, done)

                # update states (both, to avoid redundant encoding)
                current_state = observation
                latent_state = latent_observation

                # trackers
                episode_reward += reward

            rewards.append(episode_reward)

            if episode % (episodes // 100) == 0: print(
                f"\t|-- {round(episode/episodes * 100)}% (Avg. Rew. of {sum(rewards[-(episodes//100):])/(episodes//100)})")

            if save_ckpt_per and episode % save_ckpt_per == 0:  # save check point every n episodes
                res = policy.get_current_training_state()
                res["episode"] = episode  # append current episode
                save_checkpoint(res, ckpt_dir, "ckpt_{}".format(episode))

        # Last update of the agent policy
        self.policy.finish_training()


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    repr_learner = SimpleAutoencoder(4, 2, 3)
    policy = DoubleDeepQNetwork(3, 2, eps_decay=2000)
    # env = gym.make('VisualObstaclePathing-v0')  # Create VisualObstaclePathing with default values
    # size = 30
    # gym.envs.register(
    #     id='VisualObstaclePathing-v1',
    #     entry_point='src.gym_pathing.envs:ObstaclePathing',
    #     kwargs={'width': size, 'height': size,
    #             'obstacles': [[0, 18, 18, 21],
    #                           [21, 24, 10, 30]],
    #             'visual': True},
    # )
    # env = gym.make('VisualObstaclePathing-v1')

    # repr_learner = CerberusPixel(width=size,
    #                              height=size,
    #                              n_actions=len(env.action_space),
    #                              n_hidden=size)
    # policy = DoubleDeepQNetwork(size, len(env.action_space))

    # AGENT
    agent = ParallelAgent(repr_learner, policy, env)

    # TRAIN
    agent.train_agent(episodes=1000)

    # TEST
    agent.test()
    agent.env.close()
