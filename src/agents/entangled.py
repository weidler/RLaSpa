from src.policy.ddqn import DoubleDeepQNetwork
from src.policy.dqn import DeepQNetwork
from src.policy.tablebased import QTableSARSA, QTableOffPolicy
from src.representation.network.autoencoder import AutoencoderNetwork
from src.task.pathing import SimplePathing, ObstaclePathing


class EntangledAgent(object):

    def __init__(self, representation_learner, policy, environment):
        self.representation_learner = representation_learner
        self.policy = policy
        self.env = environment

    def train_representation_learner(self):
        pass

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
    # env = ObstaclePathing(30, 30,
    #                       [[0, 18, 18, 21],
    #                        [21, 24, 10, 30]]
    #                       )
    repr_learner = AutoencoderNetwork()
    policy = QTableSARSA([env.height, env.width], len(env.action_space))
    # policy = QTableOffPolicy([env.height, env.width], len(env.action_space))
    # policy = DoubleDeepQNetwork(2, len(env.action_space))
    # policy = DeepQNetwork(2, len(env.action_space))

    # AGENT
    agent = EntangledAgent(repr_learner, policy, env)

    # TRAIN
    episodes = 10000
    max_steps = 1000
    rewards = []

    for episode in range(episodes):
        done = False
        state = env.reset()
        episode_reward = 0
        steps = 0
        while not done and steps < max_steps:
            state, reward, done = agent.train_policy(state, episode)
            episode_reward += reward
            steps += 1
        rewards.append(episode_reward)

        if (episode % 10) == 0:
            print(episode, " Rewards: ", rewards[-10:])

    # Last update of the agent policy
    agent.policy.finish_training()

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
