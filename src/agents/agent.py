import abc

import torch


class _Agent(abc.ABC):

    @abc.abstractmethod
    def __init__(self, repr_learner, policy, env):
        self.env = env
        self.policy = policy
        self.representation_learner = repr_learner

    @abc.abstractmethod
    def train_agent(self, episodes: int, ckpt_to_load=None, save_ckpt_per=None):
        raise NotImplementedError

    def act(self, current_state):
        current_state = self.representation_learner.encode(current_state)
        action = self.policy.choose_action_policy(current_state)
        next_state, step_reward, env_done, _ = self.step_env(action)
        return next_state, step_reward, env_done

    def step_env(self, action):
        """
        Converts the next state to a tensor for better performance
        :param action:
        :return:
        """
        next_state, step_reward, env_done, info = self.env.step(action)
        # convert next_state to a tensor
        tensor_state = torch.Tensor(next_state).float()
        return tensor_state, step_reward, env_done, info

    def reset_env(self):
        return torch.Tensor(self.env.reset()).float()

    def test(self):
        done = False
        state = self.reset_env()
        step = 0
        total_reward = 0
        while not done:
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
