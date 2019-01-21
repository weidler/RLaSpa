import torch
from torch import optim

from src.policy.dqn import DeepQNetwork
from src.policy.network.dqn_agent import DQN
from src.policy.network.dueling_dqn_agent import DuelingDQN
from src.utils.memory.prioritized_replay_memory import PrioritizedReplayMemory
from src.utils.model_handler import update_agent_model
from src.utils.schedules import Schedule


class PrioritizedDeepQNetwork(DeepQNetwork):
    def __init__(self, num_features: int, num_actions: int, eps_calculator: Schedule, memory_eps_calculator: Schedule,
                 memory_size: int = 10000, alpha: float = 0.9, beta: float = 0.9, batch_size: int = 32,
                 learning_rate: float = 2e-3, gamma: float = 0.99, memory_delay: int = 5000,
                 representation_network: torch.nn.Module = None) -> None:
        """
        Initializes a Deep Q-Network agent with prioritized memory.

        :param num_features: Number of features that describe the environment.
        :param num_actions: Number of actions that the agent can do.
        :param eps_calculator: schedule that calculates epsilon when the memory is not used
        :param memory_eps_calculator: schedule that calculates epsilon when the memory is used
        :param memory_size: Number of plays that can be saved in the memory. Default: 10000.
        :param alpha: How much prioritization is used (0 - no prioritization, 1 - full prioritization). Default: 0.9.
        :param beta: Degree to use importance weights (0 - no corrections, 1 - full correction). Default: 0.9.
        :param batch_size: Number of memories choose to update the policy. Default: 32.
        :param learning_rate: Learning rate. Default: 2e-3.
        :param gamma: Discount factor. Default: 0.9.
        :param memory_delay: Number of steps until the memory is used.
        :param representation_network: Optional nn.Module used for the representation. Including it into the policy
            network allows full backpropagation.
        """
        super().__init__(num_features, num_actions, eps_calculator, memory_eps_calculator, memory_size, batch_size,
                         learning_rate, gamma, memory_delay, representation_network)
        self.alpha = alpha
        self.beta = beta
        self.memory = PrioritizedReplayMemory(capacity=memory_size, alpha=alpha)

    def compute_td_loss_memory(self) -> torch.tensor:
        """
        Method that computes the loss of a batch. The batch is sample for memory to take in consideration
        situations that happens before.

        :return: loss tensor
        """
        state, action, reward, next_state, done, indices, weights = self.memory.sample(self.batch_size, self.beta)

        q_values = self.model(state)

        # calculate the q-values of state with the action taken
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        # calculate the q-values of the next state
        next_q_value = self.calculate_next_q_value_memory(next_state)
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


class PrioritizedDoubleDeepQNetwork(PrioritizedDeepQNetwork):

    def __init__(self, num_features: int, num_actions: int, eps_calculator: Schedule, memory_eps_calculator: Schedule,
                 memory_size: int = 10000, alpha: float = 0.9, beta: float = 0.9, batch_size: int = 32,
                 learning_rate: float = 2e-3, gamma: float = 0.99, memory_delay: int = 5000,
                 representation_network: torch.nn.Module = None) -> None:
        # configure parent parameters
        super().__init__(num_features, num_actions, eps_calculator, memory_eps_calculator, memory_size, alpha, beta,
                         batch_size, learning_rate, gamma, memory_delay, representation_network)
        # target model needs repr network as well because otherwise copying over parameters will be non trivial
        self.target_model = DQN(num_features=num_features, num_actions=num_actions,
                                representation_network=representation_network)
        update_agent_model(current=self.model, target=self.target_model)

    def calculate_next_q_value(self, next_state: torch.Tensor) -> torch.Tensor:
        next_q_values = self.model(next_state).squeeze(0)
        next_state_value = self.target_model(next_state).squeeze(0)
        return next_state_value[torch.argmax(next_q_values)]

    def calculate_next_q_value_memory(self, next_state: torch.Tensor) -> torch.Tensor:
        next_q_values = self.model(next_state)
        next_state_value = self.target_model(next_state)
        return next_state_value.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)

    def update(self, state, action, reward, next_state, done) -> float:
        loss: torch.Tensor = super().update(state, action, reward, next_state, done)

        if self.total_steps_done % 100 == 0:
            update_agent_model(self.model, self.target_model)

        return loss

    def choose_action_policy(self, state) -> int:
        return self.target_model.act(state=state, epsilon=0)

    def finish_training(self) -> None:
        super().finish_training()
        update_agent_model(self.model, self.target_model)

    def restore_from_state(self, input) -> None:
        super().restore_from_state(input)
        self.target_model.load_state_dict(input['target_model'])

    def get_current_training_state(self):
        state = super().get_current_training_state()
        state['target_model'] = self.target_model.state_dict()
        return state


class PrioritizedDuelingDeepQNetwork(PrioritizedDoubleDeepQNetwork):
    def __init__(self, num_features: int, num_actions: int, eps_calculator: Schedule, memory_eps_calculator: Schedule,
                 memory_size: int = 10000, alpha: float = 0.9, beta: float = 0.9, batch_size: int = 32,
                 learning_rate: float = 2e-3, gamma: float = 0.99, memory_delay: int = 5000,
                 representation_network: torch.nn.Module = None) -> None:
        # configure parent parameters
        super().__init__(num_features, num_actions, eps_calculator, memory_eps_calculator, memory_size, alpha, beta,
                         batch_size, learning_rate, gamma, memory_delay, representation_network)
        self.model = DuelingDQN(num_features=num_features, num_actions=num_actions,
                                representation_network=representation_network)
        # target model needs repr network as well because otherwise copying over parameters will be non trivial
        self.target_model = DuelingDQN(num_features=num_features, num_actions=num_actions,
                                       representation_network=representation_network)
        # update the optimizer to the new model
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        update_agent_model(current=self.model, target=self.target_model)

    def calculate_next_q_value(self, next_state: torch.Tensor) -> torch.Tensor:
        next_q_values = self.target_model(next_state).squeeze(0)
        return torch.max(next_q_values)

    def calculate_next_q_value_memory(self, next_state: torch.Tensor) -> torch.Tensor:
        next_q_values = self.target_model(next_state)
        return torch.max(next_q_values, 1)[0]
