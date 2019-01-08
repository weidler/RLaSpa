import gym

from src.policy.dqn import DeepQNetwork
from src.policy.tablebased import QTableSARSA, QTableOffPolicy
from src.policy.policy import _Policy
from src.representation.learners import Janus, SimpleAutoencoder, Flatten
from src.representation.network.autoencoder import AutoencoderNetwork
from src.representation.representation import _RepresentationLearner
from src.task.pathing import SimplePathing, ObstaclePathing, VisualObstaclePathing
from src.policy.ddqn import DoubleDeepQNetwork
from src.representation.learners import SimpleAutoencoder, JanusPixel, CerberusPixel
from src.representation.representation import _RepresentationLearner
from src.task.pathing import ObstaclePathing, VisualObstaclePathing


class EntangledAgent:
    policy: _Policy
    representation_learner: _RepresentationLearner

    def __init__(self, representation_learner: _RepresentationLearner, policy: _Policy, environment):
        self.representation_learner = representation_learner
        self.policy = policy
        self.env = environment

    def train_representation_learner(self, state, action, reward, next_state):
        self.representation_learner.learn(state, action, reward, next_state)

    def train_policy(self, current_state, iteration: int):
        """
        Train the policy by performing a step in the environment and updating the policy.

        :param current_state: current state of the environment
        :param iteration: iteration number
        """
        action = self.policy.choose_action(current_state, iteration)
        next_state, step_reward, env_done, _ = self.env.step(action)

        return next_state, step_reward, env_done, action

    def act(self, current_state):
        action = self.policy.choose_action_policy(current_state)
        next_state, step_reward, env_done, _ = self.env.step(action)
        return next_state, env_done


if __name__ == "__main__":
    # env = gym.make("CartPole-v0")
    size = 30
    # env = SimplePathing(10, 10)
    env = VisualObstaclePathing(size, size,
                                [[0, 18, 18, 21],
                                 [21, 24, 10, 30]]
                                )
    # repr_learner = Flatten()
    # repr_learner = SimpleAutoencoder(4, 2, 4)
    repr_learner = CerberusPixel(width=size,
                              height=size,
                              n_actions=len(env.action_space), # this is not the number of possible actions, but the length of the action itself
                              n_hidden=size)
    # NOTE the output of the representation learner is the input of the network
    policy = DoubleDeepQNetwork(size, len(env.action_space))
    # policy = QTableSARSA([env.height, env.width], len(env.action_space))
    # policy = QTableOffPolicy([env.height, env.width], len(env.action_space))
    # policy = DoubleDeepQNetwork(2, len(env.action_space))
    # policy = DeepQNetwork(900, 2)
    # policy = DeepQNetwork(900, len(env.action_space))
    # policy = DeepQNetwork(env.observation_space.shape[0], env.action_space.n)

    # AGENT
    agent = EntangledAgent(repr_learner, policy, env)

    # TRAIN
    episodes = 10000
    max_steps = 300
    rewards = []

    for episode in range(episodes):
        done = False
        state_original = env.reset()
        state = agent.representation_learner.encode(state_original)
        episode_reward = 0
        steps = 0
        while not done and steps < max_steps:
            next_state_original, reward, done, action = agent.train_policy(state.tolist(), episode)
            next_state = agent.representation_learner.encode(next_state_original)
            agent.train_representation_learner(state=state_original, next_state=next_state_original, reward=reward, action=action)

            # this needs to be here after the next state was encoded by the repr_learner
            agent.policy.update(state, action, reward, next_state, done)

            episode_reward += reward
            steps += 1
            state = next_state
            state_original = next_state_original

        rewards.append(episode_reward)

        if (episode % 10) == 0:
            print(episode, "Average Rewards: ", sum(rewards[-10:]) / 10)

    # Last update of the agent policy
    agent.policy.finish_training()

    # TEST
    done = False
    state = env.reset()
    while not done and max_steps > 0:
        state, done = agent.act(state)
        max_steps -= 1

    env.show_breadcrumbs = True
    print(env.target_coords)
    print(env)
