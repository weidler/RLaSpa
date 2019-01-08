import gym

from src.policy.ddqn import DoubleDeepQNetwork
from src.policy.policy import _Policy
from src.representation.learners import SimpleAutoencoder, JanusPixel, CerberusPixel
from src.representation.representation import _RepresentationLearner
from src.task.pathing import VisualObstaclePathing


class ParallelAgent:
    policy: _Policy
    representation_learner: _RepresentationLearner

    def __init__(self, representation_learner: _RepresentationLearner, policy: _Policy, environment):
        self.representation_learner = representation_learner
        self.policy = policy
        self.env = environment

    # REINFORCEMENT LEARNING #

    def train_agent(self, episodes: int, max_episode_length=1000):
        print("Starting parallel training process.")
        rewards = []
        for episode in range(episodes):
            done = False
            current_state = env.reset()
            latent_state = self.representation_learner.encode(current_state)
            episode_reward = 0
            steps = 0
            while not done and steps < max_episode_length:
                # choose action
                action = self.policy.choose_action(latent_state, episode)

                # step and observe
                observation, reward, done, _ = self.env.step(action)
                latent_observation = self.representation_learner.encode(observation)

                # train the REPRESENTATION learner
                self.representation_learner.learn(state=current_state, next_state=observation, reward=reward,
                                                  action=action)

                # train the POLICY
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

    # TESTING #

    def act(self, current_state):
        current_state = self.representation_learner.encode(current_state)
        action = self.policy.choose_action_policy(current_state)
        next_state, step_reward, env_done, _ = self.env.step(action)
        return next_state, step_reward, env_done

    def test(self, max_episode_length=1000):
        done = False
        state = env.reset()
        step = 0
        total_reward = 0
        while not done and step < max_episode_length:
            env.render()
            state, reward, done = agent.act(state)
            step += 1
            total_reward += reward

        print(f"Episode finished after {step} steps with total reward of {total_reward}.")


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    repr_learner = SimpleAutoencoder(4, 2, 3)
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
