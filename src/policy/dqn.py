import torch
from torch import optim
from torch.nn import functional as F

from src.policy.network.dqn_agent import DQN
from src.policy.network.dueling_dqn_agent import DuelingDQN
from src.policy.policy import _Policy
from src.utils.memory.replay_memory import ReplayMemory
from src.utils.model_handler import update_agent_model
from src.utils.schedules import Schedule


class DeepQNetwork(_Policy):

    def __init__(self, num_features: int, num_actions: int, eps_calculator: Schedule, memory_eps_calculator: Schedule,
                 memory_size: int = 10000, batch_size: int = 32, learning_rate: float = 2e-3, gamma: float = 0.99,
                 memory_delay: int = 5000, representation_network: torch.nn.Module = None) -> None:
        """
        Initializes a Deep Q-Network agent

        :param num_features: Number of features that describe the environment.
        :param num_actions: Number of actions that the agent can do.
        :param eps_calculator: schedule that calculates epsilon when the memory is not used
        :param memory_eps_calculator: schedule that calculates epsilon when the memory is used
        :param memory_size: Number of plays that can be saved in the memory. Default: 10000.
        :param batch_size: Number of memories choose to update the policy. Default: 32.
        :param learning_rate: Learning rate. Default: 2e-3
        :param gamma: Discount factor. Default: 0.9
        :param memory_delay: Number of steps until the memory is used.
        :param representation_network: Optional nn.Module used for the representation. Including it into the policy
        network allows full backpropagation.
        """
        if num_features is None:
            raise ValueError('Number of features not included.')
        if num_actions is None:
            raise ValueError('Number of action not included.')
        if memory_eps_calculator is None:
            raise ValueError('The epsilon calculator for the memory cannot be null.')
        if eps_calculator is None and memory_delay != 0:
            raise ValueError('The epsilon calculator cannot be null if the memory_delay is not 0.')
        self.gamma = torch.tensor(gamma).float()
        self.total_steps_done = 0  # Counter to control the memory activation
        self.batch_size = batch_size
        self.use_memory = False
        self.memory_delay = memory_delay
        self.memory = ReplayMemory(capacity=memory_size)
        self.model = DQN(num_features=num_features, num_actions=num_actions,
                         representation_network=representation_network)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.epsilon_calculator = eps_calculator
        self.memory_epsilon_calculator = memory_eps_calculator

    def calculate_next_q_value(self, next_state: torch.Tensor) -> torch.Tensor:
        """
        Method that calculates the next Q value given the next state. Used to calculate the loss
        when the memory is not used. This method handles only one state.

        :param next_state: state of the environment after acting
        :return: estimation of the next state q value
        """
        next_q_values = self.model(next_state).squeeze(0)
        return torch.max(next_q_values)

    def calculate_next_q_value_memory(self, next_state: torch.Tensor) -> torch.Tensor:
        """
        Method that calculates the next Q value given the next state. Used to calculate the loss when
        the memory is in used. This method handle a list of states.

        :param next_state: list of states of the environment after acting
        :return: estimation of the next state q value
        """
        next_q_values = self.model(next_state)
        return torch.max(next_q_values, 1)[0]

    def compute_td_loss(self, state: torch.tensor, action: torch.tensor, reward: torch.tensor, next_state: torch.tensor,
                        done: torch.tensor) -> torch.tensor:
        """
        Method to compute the loss for a given iteration, in general is used when the memory mechanism is off

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

        # making batches of one
        state = state.unsqueeze(0)
        next_state = next_state.unsqueeze(0)

        q_values = self.model(state).squeeze(0)

        # calculate the q-values of state with the action taken
        q_value = q_values[action]
        # calculate the q-values of the next state
        next_q_value = self.calculate_next_q_value(next_state)
        # 0 if next state was 0
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = (q_value - expected_q_value.detach()).pow(2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def compute_td_loss_memory(self) -> torch.Tensor:
        """
        Method that computes the loss of a batch. The batch is sample for memory to take in consideration
        situations that happens before.

        :return: loss tensor
        """
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)

        q_values = self.model(state)

        # calculate the q-values of state with the action taken
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        # calculate the q-values of the next state
        next_q_value = self.calculate_next_q_value_memory(next_state)
        # 0 if next state was 0
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = (q_value - expected_q_value.detach()).pow(2).mean()
        # loss = F.smooth_l1_loss(q_value, expected_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def update(self, state, action, reward, next_state, done) -> float:
        loss: torch.Tensor = None
        self.total_steps_done += 1
        # memory or not memory
        if self.total_steps_done >= self.memory_delay:
            # save in memory
            self.memory.push(state, action, reward, next_state, done)
            # when saved plays are greater than the batch size calculate losses
            if len(self.memory) > self.batch_size:
                loss = self.compute_td_loss_memory()

            if self.total_steps_done == self.memory_delay:
                print("\tPolicy-DQN begins memorizing now.")
        else:
            loss = self.compute_td_loss(state, action, reward, next_state, done)

        return 0 if loss is None else loss.item()

    def choose_action(self, state) -> int:
        if not self.use_memory and self.total_steps_done >= self.memory_delay:
            self.use_memory = True
        if self.use_memory:
            epsilon = self.memory_epsilon_calculator.value(self.total_steps_done - self.memory_delay)
        else:
            epsilon = self.epsilon_calculator.value(self.total_steps_done)
        return self.model.act(state=state, epsilon=epsilon)

    def choose_action_policy(self, state) -> int:
        return self.model.act(state=state, epsilon=0)

    def finish_training(self) -> None:
        self.total_steps_done = 0

    def restore_from_state(self, input) -> None:
        self.model.load_state_dict(input['model'])
        self.optimizer.load_state_dict(input['optimizer'])
        self.total_steps_done = input['total_steps']

    def get_current_training_state(self):
        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps_done,
        }


class DoubleDeepQNetwork(DeepQNetwork):
    def __init__(self, num_features: int, num_actions: int, eps_calculator: Schedule, memory_eps_calculator: Schedule,
                 memory_size: int = 10000, batch_size: int = 32, learning_rate: float = 2e-3, gamma: float = 0.99,
                 memory_delay: int = 5000, representation_network: torch.nn.Module = None) -> None:
        # configure parent parameters
        super().__init__(num_features, num_actions, eps_calculator, memory_eps_calculator, memory_size, batch_size,
                         learning_rate, gamma, memory_delay, representation_network)
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


class DuelingDeepQNetwork(DoubleDeepQNetwork):
    def __init__(self, num_features: int, num_actions: int, eps_calculator: Schedule, memory_eps_calculator: Schedule,
                 memory_size: int = 10000, batch_size: int = 32, learning_rate: float = 2e-3, gamma: float = 0.99,
                 memory_delay: int = 5000, representation_network: torch.nn.Module = None) -> None:
        # configure parent parameters
        super().__init__(num_features, num_actions, eps_calculator, memory_eps_calculator, memory_size, batch_size,
                         learning_rate, gamma, memory_delay, representation_network)
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
