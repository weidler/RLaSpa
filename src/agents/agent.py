import abc


class _Agent(abc.ABC):

    @abc.abstractmethod
    def __init__(self, repr_learner, policy, env):
        self.env = env
        self.policy = policy
        self.representation_learner = repr_learner

    @abc.abstractmethod
    def train_agent(self, episodes: int, max_episode_length=1000, ckpt_to_load=None, save_ckpt_per=None):
        raise NotImplementedError

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
            self.env.render()

        print(f"Tested episode took {step} steps and gathered a reward of {total_reward}.")

    def get_config_name(self):
        return "_".join(
            [self.__class__.__name__,
             self.env.__class__.__name__,
             self.representation_learner.__class__.__name__,
             self.policy.__class__.__name__])
