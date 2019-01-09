import argparse

import gym
import torch
from torch import optim

from src.policy.network.dqn_agent import DQN
from src.utils.memory.prioritized_replay_memory import PrioritizedReplayMemory
from src.utils.model_handler import update_agent_model, save_model
from src.utils.schedules import ExponentialSchedule


def compute_td_loss(batch_size: int, beta: float):
    """
    Method that computes the loss of a batch. The batch is sample for memory to take in consideration
    situations that happens before.

    :param batch_size: number of plays that will be used
    :param beta: degree to use importance weights (0 - no corrections, 1 - full correction)
    :return: loss for the whole batch
    """
    state, action, reward, next_state, done, indices, weights = memory.sample(batch_size, beta)

    state = torch.tensor(state, dtype=torch.float32)
    next_state = torch.tensor(next_state, dtype=torch.float32)
    action = torch.tensor(action, dtype=torch.long)
    reward = torch.tensor(reward, dtype=torch.float32)
    done = torch.tensor(done, dtype=torch.float32)
    weights = torch.tensor(weights, dtype=torch.float32)

    q_values = current_model(state)
    next_q_values = current_model(next_state)
    next_state_value = target_model(next_state)

    # calculate the q-values of state with the action taken
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    # calculate the state value using the target model
    next_q_value = next_state_value.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    # 0 if next state was 0
    expected_q_value = reward + args.gamma * next_q_value * (1 - done)

    loss = (q_value - expected_q_value.detach()).pow(2) * weights
    prios = loss + 1e-5
    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    memory.update_priorities(indices, prios.data.cpu().numpy())
    optimizer.step()

    return loss


def train(iterations: int, batch_size: int):
    state = env.reset()
    losses = []
    all_rewards = []
    episode_reward = 0
    for iteration in range(1, iterations + 1):
        epsilon = epsilon_calculator.value(time_step=iteration)
        action = current_model.act(state=state, epsilon=epsilon)

        next_state, reward, done, _ = env.step(action)
        memory.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if done:
            state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0

        if len(memory) > batch_size:
            # when saved plays are greater than the batch size calculate losses
            loss = compute_td_loss(batch_size=batch_size, beta=args.beta)
            losses.append(loss.item())

        if iteration % 100 == 0:
            update_agent_model(current=current_model, target=target_model)

        if iteration % 200 == 0:
            print('Iteration: {0}'.format(iteration))
            print('Rewards: {0}'.format(all_rewards[-9:]))
    update_agent_model(current=current_model, target=target_model)


def play(iterations: int, render=True):
    state = env.reset()
    rewards = []
    episode_reward = 0
    for iteration in range(1, iterations + 1):
        if render:
            env.render()
        action = target_model.act(state=state, epsilon=0)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        state = next_state
        if done:
            state = env.reset()
            rewards.append(episode_reward)
            episode_reward = 0
    if render:
        env.close()
    print('End of game. Final rewards {0}'.format(rewards))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DDQN agent execution')
    parser.add_argument('--env', type=str, metavar='E', default='CartPole-v0', help='GYM environment')
    parser.add_argument('--init_eps', type=float, metavar='I', default=1.0, help='Initial epsilon')
    parser.add_argument('--min_eps', type=float, metavar='M', default=0.01, help='Minimum epsilon')
    parser.add_argument('--eps_decay', type=int, metavar='D', default=5000, help='Epsilon decay')
    parser.add_argument('--gamma', type=int, metavar='G', default=0.99, help='Gamma')
    parser.add_argument('--memory_size', type=int, metavar='S', default=10000, help='Memory size')
    parser.add_argument('--alpha', type=float, metavar='A', default=0.8,
                        help='How much prioritization is used (0 - no prioritization, 1 - full prioritization)')
    parser.add_argument('--beta', type=float, metavar='B', default=0.8,
                        help='Degree to use importance weights (0 - no corrections, 1 - full correction)')
    parser.add_argument('--batch_size', type=int, metavar='BS', default=32, help='Batch size')
    parser.add_argument('--iterations', type=int, metavar='IT', default=30000, help='Training iterations')
    args = parser.parse_args()
    # Environment information extraction
    env = gym.make(args.env)
    number_of_observations = env.observation_space.shape[0]
    number_of_actions = env.action_space.n
    # Agent creation and configuration
    current_model = DQN(num_features=number_of_observations, num_actions=number_of_actions)
    target_model = DQN(num_features=number_of_observations, num_actions=number_of_actions)
    update_agent_model(current=current_model, target=target_model)
    optimizer = optim.Adam(current_model.parameters())
    memory = PrioritizedReplayMemory(capacity=args.memory_size, alpha=args.alpha)
    epsilon_calculator = ExponentialSchedule(initial_p=args.init_eps, min_p=args.min_eps, decay=args.eps_decay)
    # Training
    train(iterations=args.iterations, batch_size=args.batch_size)
    play(iterations=10000, render=True)
    save_model(target_model, "../../models/ddqn.model")
