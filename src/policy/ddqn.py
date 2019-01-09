import torch
from torch import optim

from src.policy.network.dqn_agent import DQN
from src.policy.network.dueling_dqn_agent import DuelingDQN
from src.policy.policy import _Policy
from src.utils.memory.prioritized_replay_memory import PrioritizedReplayMemory
from src.utils.model_handler import update_agent_model
from src.utils.schedules import ExponentialSchedule


class DoubleDeepQNetwork(_Policy):

    def __init__(self, num_features: int, num_actions: int, memory_size=10000, alpha=0.9, beta=0.9, batch_size=32,
                 learning_rate=2e-3, gamma=0.9, init_eps=1.0, min_eps=0.01, eps_decay=500, memory_delay=5000):
        """
        Initializes a Double Deep Q-Network agent with prioritized memory.

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
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.total_steps_done = 0
        self.batch_size = batch_size
        self.memory_delay = memory_delay
        self.current_model = DQN(num_features=num_features, num_actions=num_actions)
        self.target_model = DQN(num_features=num_features, num_actions=num_actions)
        self.optimizer = optim.Adam(self.current_model.parameters(), lr=learning_rate)
        self.memory = PrioritizedReplayMemory(capacity=memory_size, alpha=alpha)
        self.epsilon_calculator = ExponentialSchedule(initial_p=init_eps, min_p=min_eps, decay=eps_decay)
        update_agent_model(current=self.current_model, target=self.target_model)

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

        q_values = self.current_model(state)
        next_q_values = self.current_model(next_state)
        next_state_value = self.target_model(next_state)

        # calculate the q-values of state with the action taken
        q_value = q_values[action]
        # calculate the q-values of the next state
        next_q_value = next_state_value[torch.argmax(next_q_values)]
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
        state, action, reward, next_state, done, indices, weights = self.memory.sample(self.batch_size, self.beta)

        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32)

        q_values = self.current_model(state)
        next_q_values = self.current_model(next_state)
        next_state_value = self.target_model(next_state)

        # calculate the q-values of state with the action taken
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        # calculate the state value using the target model
        next_q_value = next_state_value.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        # 0 if next state was 0
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = (q_value - expected_q_value.detach()).pow(2) * weights
        prios = loss + 1e-5
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.memory.update_priorities(indices, prios.data.cpu().numpy())
        self.optimizer.step()

    def update(self, state, action, reward, next_state, done) -> None:
        self.total_steps_done += 1
        if self.total_steps_done > self.memory_delay:
            self.memory.push(state, action, reward, next_state, done)
            if len(self.memory) > self.batch_size:
                # when saved plays are greater than the batch size calculate losses
                self.compute_td_loss_memory()
        else:
            self.compute_td_loss(state, action, reward, next_state, done)
            if self.total_steps_done == self.memory_delay: print("\tPolicy-DQN begins memorizing now.")
        if self.total_steps_done % 100:
            update_agent_model(self.current_model, self.target_model)

    def choose_action(self, state) -> int:
        epsilon = self.epsilon_calculator.value(self.total_steps_done)
        return self.current_model.act(state=state, epsilon=epsilon)

    def choose_action_policy(self, state) -> int:
        return self.current_model.act(state=state, epsilon=0)

    def finish_training(self) -> None:
        self.total_steps_done = 0
        update_agent_model(self.current_model, self.target_model)


class DuelingDeepQNetwork(DoubleDeepQNetwork):
    def __init__(self, num_features: int, num_actions: int, memory_size=10000, alpha=0.9, beta=0.9, batch_size=32,
                 learning_rate=2e-3, gamma=0.9, init_eps=1.0, min_eps=0.01, eps_decay=5000):
        """
        Initializes a Double Deep Q-Network agent with prioritized memory.

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
        """
        super().__init__(num_features, num_actions, memory_size, alpha, beta, batch_size, learning_rate, gamma,
                         init_eps, min_eps, eps_decay)
        self.current_model = DuelingDQN(num_features=num_features, num_actions=num_actions)
        self.target_model = DuelingDQN(num_features=num_features, num_actions=num_actions)
        update_agent_model(current=self.current_model, target=self.target_model)

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

        q_values = self.current_model(state)
        next_q_values = self.target_model(next_state)

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
        state, action, reward, next_state, done, indices, weights = self.memory.sample(self.batch_size, self.beta)

        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32)

        q_values = self.current_model(state)
        next_q_values = self.target_model(next_state)

        # calculate the q-values of state with the action taken
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        # calculate the state value using the target model
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
