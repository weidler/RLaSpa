import torch
from torch import optim

from src.agents.dqn_agent import DQN
from src.policy.policy import _Policy
from src.utils.memory.prioritized_replay_memory import PrioritizedReplayMemory
from src.utils.memory.replay_memory import ReplayMemory
from src.utils.schedules import ExponentialSchedule


class DeepQNetwork(_Policy):

    def __init__(self, num_features: int, num_actions: int, memory_size=10000, batch_size=32, learning_rate=2e-3,
                 gamma=0.99, init_eps=1.0, min_eps=0.01, eps_decay=500, memory_delay=5000) -> None:
        """
        Initializes a Deep Q-Network agent

        :param num_features: Number of features that describe the environment.
        :param num_actions: Number of actions that the agent can do.
        :param memory_size: Number of plays that can be saved in the memory. Default: 10000.
        :param batch_size: Number of memories choose to update the policy. Default: 32.
        :param learning_rate: Learning rate. Default: 2e-3
        :param gamma: Discount factor. Default: 0.9
        :param init_eps: Initial epsilon. Default:1.0
        :param min_eps: Minimal epsilon. Default: 0.01
        :param eps_decay: Number of steps for epsilon convergence to the minimal value.
        :param memory_delay: Number of steps until the memory is used.
        """
        self.gamma = gamma
        self.num_policy_updates = 0  # Counter to control the memory activation
        self.batch_size = batch_size
        self.memory_delay = memory_delay
        self.memory = ReplayMemory(capacity=memory_size)
        self.model = DQN(num_features=num_features, num_actions=num_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.epsilon_calculator = ExponentialSchedule(initial_p=init_eps, min_p=min_eps, decay=eps_decay)

    def compute_td_loss(self, state, action, reward, next_state, done) -> None:
        """
        Method to compute the loss for a given iteration

        :param state: initial state
        :param action: action taken
        :param reward: reward received
        :param next_state: state after acting
        :param done: flag that indicates if the episode has finished
        """
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        q_values = self.model(state)
        next_q_values = self.model(next_state)

        # calculate the q-values of state with the action taken
        q_value = q_values[action]
        # calculate the q-values of the next state
        next_q_value = torch.max(next_q_values)
        # 0 if next state was 0
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = (q_value - expected_q_value.detach()).pow(2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def compute_td_loss_memory(self) -> None:
        """
        Method that computes the loss of a batch. The batch is sample for memory to take in consideration
        situations that happens before.
        """
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        q_values = self.model(state)
        next_q_values = self.model(next_state)

        # calculate the q-values of state with the action taken
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        # calculate the q-values of the next state
        next_q_value = torch.max(next_q_values, 1)[0]
        # 0 if next state was 0
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update(self, state, action, reward, next_state, done) -> None:
        self.num_policy_updates += 1
        if self.num_policy_updates > self.memory_delay:
            self.memory.push(state, action, reward, next_state, done)
            if len(self.memory) > self.batch_size:
                # when saved plays are greater than the batch size calculate losses
                self.compute_td_loss_memory()
        else:
            self.compute_td_loss(state, action, reward, next_state, done)

    def choose_action(self, state, iteration: int) -> int:
        epsilon = self.epsilon_calculator.value(iteration)
        return self.model.act(state=state, epsilon=epsilon)

    def choose_action_policy(self, state) -> int:
        return self.model.act(state=state, epsilon=0)

    def finish_training(self) -> None:
        pass


class PrioritizedDeepQNetwork(DeepQNetwork):
    def __init__(self, num_features: int, num_actions: int, memory_size=10000, alpha=0.9, beta=0.9, batch_size=32,
                 learning_rate=2e-3, gamma=0.9, init_eps=1.0, min_eps=0.01, eps_decay=500, memory_delay=5000):
        """
        Initializes a Deep Q-Network agent with prioritized memory.

        :param num_features: Number of features that describe the environment.
        :param num_actions: Number of actions that the agent can do.
        :param memory_size: Number of plays that can be saved in the memory. Default: 10000.
        :param batch_size: Number of memories choose to update the policy. Default: 32.
        :param learning_rate: Learning rate. Default: 2e-3.
        :param gamma: Discount factor. Default: 0.9.
        :param init_eps: Initial epsilon. Default:1.0.
        :param min_eps: Minimal epsilon. Default: 0.01.
        :param eps_decay: Number of steps for epsilon convergence to the minimal value.
        :param alpha: How much prioritization is used (0 - no prioritization, 1 - full prioritization). Default: 0.9.
        :param beta: Degree to use importance weights (0 - no corrections, 1 - full correction). Default: 0.9.
        :param memory_delay: Number of steps until the memory is used.
        """
        super().__init__(num_features, num_actions, memory_size, batch_size, learning_rate, gamma, init_eps, min_eps,
                         eps_decay, memory_delay)
        self.alpha = alpha
        self.beta = beta
        self.memory = PrioritizedReplayMemory(capacity=memory_size, alpha=alpha)

    def compute_td_loss_memory(self) -> None:
        """
        Method that computes the loss of a batch. The batch is sample for memory to take in consideration
        situations that happens before.
        """
        state, action, reward, next_state, done, indices, weights = self.memory.sample(self.batch_size, self.beta)

        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32)

        q_values = self.model(state)
        next_q_values = self.model(next_state)

        # calculate the q-values of state with the action taken
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        # calculate the q-values of the next state
        next_q_value = torch.max(next_q_values, 1)[0]
        # 0 if next state was 0
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = (q_value - expected_q_value.detach()).pow(2) * weights
        prios = loss + 1e-5
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.memory.update_priorities(indices, prios.data.cpu().numpy())
        self.optimizer.step()