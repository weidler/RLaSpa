from src.policy.tablebased import QTableOffPolicy, QTableSARSA
from src.representation.autoencoder import Autoencoder
from src.task.pathing import SimplePathing


class EntangledAgent(object):

    def __init__(self, representation_learner, policy, environment):
        self.representation_learner = representation_learner
        self.policy = policy
        self.env = environment

    def train_representation_learner(self):
        pass

    def train_policy(self, current_state, action=None):
        """ Train the policy by performing a step in the environment and updating the policy.

        :param current_state:   the state that the agent is currently in
        :param action:          (default None) only specify if agent learns on-policy and needs next action
        """
        if action is None:
            action = self.policy.choose_action(current_state)

        # observe
        observation, reward, done = self.env.step(action)
        next_action = self.policy.choose_action(current_state)

        # update policy
        self.policy.update(current_state, action, reward, observation, next_action)

        return observation, next_action, done

    def act(self, state):
        action = self.policy.choose_action(state)
        observation, reward, done = self.env.step(action)

        return observation, done


if __name__ == "__main__":
    env = SimplePathing(10, 10)
    repr_learner = Autoencoder()
    policy = QTableSARSA([env.height, env.width], len(env.action_space))

    # AGENT
    agent = EntangledAgent(repr_learner, policy, env)

    # TRAIN
    episodes = 10000
    for episode in range(episodes):
        done = False
        state = env.reset()
        action = None
        while not done:
            state, action, done = agent.train_policy(state)

        if (episode % 100) == 0: print(episode)

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
