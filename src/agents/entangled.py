from src.policy.ddqn import DoubleDeepQNetwork
from src.policy.policy import _Policy
from src.policy.tablebased import QTableSARSA, QTableOffPolicy
from src.policy.tablebased import QTableSARSA
from src.representation.network.autoencoder import AutoencoderNetwork
from src.representation.representation import _RepresentationLearner
from src.task.pathing import SimplePathing


class EntangledAgent:

    policy: _Policy
    representation_learner: _RepresentationLearner

    def __init__(self, representation_learner: _RepresentationLearner, policy: _Policy, environment):
        self.representation_learner = representation_learner
        self.policy = policy
        self.env = environment

    def train_representation_learner(self, state):
        self.representation_learner.learn(state)

    def train_policy(self, current_state, iteration: int):
        """
        Train the policy by performing a step in the environment and updating the policy.

        :param current_state: current state of the environment
        :param iteration: iteration number
        """
        action = self.policy.choose_action(current_state, iteration)
        next_state, step_reward, env_done = self.env.step(action)

        self.policy.update(current_state, action, step_reward, next_state, env_done)

        return next_state, step_reward, env_done

    def act(self, current_state):
        action = self.policy.choose_action_policy(current_state)
        next_state, step_reward, env_done = self.env.step(action)
        return next_state, env_done


if __name__ == "__main__":
    env = SimplePathing(10, 10)
    repr_learner = AutoencoderNetwork()
    # policy = QTableSARSA([env.height, env.width], len(env.action_space))
    # policy = QTableOffPolicy([env.height, env.width], len(env.action_space))
    policy = DoubleDeepQNetwork(2, len(env.action_space))
    # policy = QTableSARSA([env.height, env.width], len(env.action_space))

    # AGENT
    agent = EntangledAgent(repr_learner, policy, env)

    # TRAIN
    episodes = 30000

    rewards = []
    episode_reward = 0

    state = env.reset()
    for episode in range(episodes):
        state, reward, done = agent.train_policy(state, episode)
        episode_reward += reward

        if done:
            state = env.reset()
            rewards.append(episode_reward)
            episode_reward = 0

        if (episode % 100) == 0:
            print(episode)

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
