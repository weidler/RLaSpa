import torch
from torch import optim

from src.policy.network.dqn_agent import DQN
from src.policy.network.dueling_dqn_agent import DuelingDQN
from src.policy.policy import _Policy
from src.utils.memory.prioritized_replay_memory import PrioritizedReplayMemory
from src.utils.model_handler import update_agent_model
from src.utils.schedules import ExponentialSchedule, LinearSchedule


class DoubleDeepQNetwork(_Policy):

    def __init__(self, num_features: int, num_actions: int, memory_size: int = 10000, alpha: float = 0.9,
                 beta: float = 0.9, batch_size: int = 32, learning_rate: float = 2e-3, gamma: float = 0.99,
                 init_eps: float = 1.0, min_eps=0.01, eps_decay=500, per_init_eps_memory: int = 0.8,
                 memory_delay: int = 5000, representation_network: torch.nn.Module = None) -> None:
        """
        Initializes a Double Deep Q-Network agent with prioritized memory.

        :param num_features: Number of features that describe the environment.
        :param num_actions: Number of actions that the agent can do.
        :param memory_size: Number of plays that can be saved in the memory. Default: 10000.
        :param alpha: How much prioritization is used (0 - no prioritization, 1 - full prioritization). Default: 0.9.
        :param beta: Degree to use importance weights (0 - no corrections, 1 - full correction). Default: 0.9.
        :param batch_size: Number of memories choose to update the policy. Default: 32.
        :param learning_rate: Learning rate. Default: 2e-3.
        :param gamma: Discount factor. Default: 0.9.
        :param init_eps: Initial epsilon. Default:1.0.
        :param min_eps: Minimal epsilon. Default: 0.01.
        :param eps_decay: Number of steps for epsilon convergence to the minimal value since the use of the memory.
        Default: 500
        :param per_init_eps_memory: percentage of the initial epsilon that will remain when
        the memory starts to be used. Default: 0.8
        :param memory_delay: Number of steps until the memory is used.
        :param representation_network: Optional nn.Module used for the representation. Including it into the policy
        network allows full backpropagation.
        """
        self.beta = beta
        self.alpha = alpha
        self.total_steps_done = 0
        self.batch_size = batch_size
        self.memory_delay = memory_delay
        self.gamma = torch.tensor(gamma, dtype=torch.float32)
        self.current_model = DQN(num_features=num_features, num_actions=num_actions,
                                 representation_network=representation_network)
        # target model needs repr network as well because otherwise copying over parameters will be non trivial
        self.target_model = DQN(num_features=num_features, num_actions=num_actions,
                                representation_network=representation_network)
        self.optimizer = optim.Adam(self.current_model.parameters(), lr=learning_rate)
        self.memory = PrioritizedReplayMemory(capacity=memory_size, alpha=alpha)
        self.epsilon_calculator = LinearSchedule(schedule_timesteps=self.memory_delay, initial_p=init_eps,
                                                 final_p=init_eps * per_init_eps_memory)
        self.memory_epsilon_calculator = ExponentialSchedule(initial_p=init_eps * per_init_eps_memory, min_p=min_eps,
                                                             decay=eps_decay)
        update_agent_model(current=self.current_model, target=self.target_model)

    def compute_td_loss(self, state, action, reward, next_state, done) -> torch.tensor:
        """
        Method to compute the loss for a given iteration

        :param state: initial state
        :param action: action taken
        :param reward: reward received
        :param next_state: state after acting
        :param done: flag that indicates if the episode has finished
        :return: loss tensor
        """
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

        return loss

    def compute_td_loss_memory(self) -> torch.tensor:
        """
        Method that computes the loss of a batch. The batch is sample for memory to take in consideration
        situations that happens before.

        :return: loss tensor
        """

        state, action, reward, next_state, done, indices, weights = self.memory.sample(self.batch_size, self.beta)

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

        return loss

    def update(self, state, action, reward, next_state, done) -> float:
        self.total_steps_done += 1

        loss = None
        if self.total_steps_done > self.memory_delay:
            self.memory.push(state, action, reward, next_state, done)
            if len(self.memory) > self.batch_size:
                # when saved plays are greater than the batch size calculate losses
                loss = self.compute_td_loss_memory()
        else:
            loss = self.compute_td_loss(state, action, reward, next_state, done)
            if self.total_steps_done == self.memory_delay:
                print("\tPolicy-DQN begins memorizing now.")
        if self.total_steps_done % 100:
            update_agent_model(self.current_model, self.target_model)

        return 0 if loss is None else loss.item()

    def choose_action(self, state) -> int:
        if self.total_steps_done > self.memory_delay:
            epsilon = self.memory_epsilon_calculator.value(self.total_steps_done - self.memory_delay)
        else:
            epsilon = self.epsilon_calculator.value(self.total_steps_done)
        return self.current_model.act(state=state, epsilon=epsilon)

    def choose_action_policy(self, state) -> int:
        return self.current_model.act(state=state, epsilon=-1)

    def finish_training(self) -> None:
        self.total_steps_done = 0
        update_agent_model(self.current_model, self.target_model)

    def restore_from_state(self, input) -> None:
        self.current_model.load_state_dict(input['current_model'])
        self.target_model.load_state_dict(input['target_model'])
        self.optimizer.load_state_dict(input['optimizer'])
        self.total_steps_done = input['total_steps']
        self.memory = input['memory']

    def get_current_training_state(self):
        return {
            'current_model': self.current_model.state_dict(),
            'target_model': self.target_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps_done,
            'memory': self.memory
        }


class DuelingDeepQNetwork(DoubleDeepQNetwork):
    def __init__(self, num_features: int, num_actions: int, memory_size: int = 10000, alpha: float = 0.9,
                 beta: float = 0.9, batch_size: int = 32, learning_rate: float = 2e-3, gamma: float = 0.99,
                 init_eps: float = 1.0, min_eps=0.01, eps_decay=500, per_init_eps_memory: int = 0.8,
                 memory_delay: int = 5000, representation_network: torch.nn.Module = None) -> None:
        """
        Initializes a Double Deep Q-Network agent with prioritized memory.

        :param num_features: Number of features that describe the environment.
        :param num_actions: Number of actions that the agent can do.
        :param memory_size: Number of plays that can be saved in the memory. Default: 10000.
        :param alpha: How much prioritization is used (0 - no prioritization, 1 - full prioritization). Default: 0.9.
        :param beta: Degree to use importance weights (0 - no corrections, 1 - full correction). Default: 0.9.
        :param batch_size: Number of memories choose to update the policy. Default: 32.
        :param learning_rate: Learning rate. Default: 2e-3.
        :param gamma: Discount factor. Default: 0.9.
        :param init_eps: Initial epsilon. Default:1.0.
        :param min_eps: Minimal epsilon. Default: 0.01.
        :param eps_decay: Number of steps for epsilon convergence to the minimal value since the use of the memory.
        Default: 500
        :param per_init_eps_memory: percentage of the initial epsilon that will remain when
        the memory starts to be used. Default: 0.8
        :param memory_delay: Number of steps until the memory is used.
        :param representation_network: Optional nn.Module used for the representation. Including it into the policy
        network allows full backpropagation.
        """
        super().__init__(num_features, num_actions, memory_size, alpha, beta, batch_size, learning_rate, gamma,
                         init_eps, min_eps, eps_decay, per_init_eps_memory, memory_delay, representation_network)
        self.current_model = DuelingDQN(num_features=num_features, num_actions=num_actions,
                                        representation_network=representation_network)
        self.target_model = DuelingDQN(num_features=num_features, num_actions=num_actions,
                                       representation_network=representation_network)
        update_agent_model(current=self.current_model, target=self.target_model)

    def compute_td_loss(self, state, action, reward, next_state, done) -> torch.tensor:
        """
        Method to compute the loss for a given iteration

        :param state: initial state
        :param action: action taken
        :param reward: reward received
        :param next_state: state after acting
        :param done: flag that indicates if the episode has finished
        :return: loss tensor
        """
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

        return loss

    def compute_td_loss_memory(self) -> torch.tensor:
        """
        Method that computes the loss of a batch. The batch is sample for memory to take in consideration
        situations that happens before.

        :return: loss tensor
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

        return loss
