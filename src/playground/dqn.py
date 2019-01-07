import argparse

import gym
import torch
from torch import optim

from src.agents.dqn_agent import DQN
from src.utils.memory.replay_memory import ReplayMemory
from src.utils.schedules import ExponentialSchedule


def compute_td_loss(batch_size: int):
    """
    Method that computes the loss of a batch. The batch is sample for memory to take in consideration
    situations that happens before.

    :param batch_size: number of plays that will be used
    :return: loss for the whole batch
    """
    state, action, reward, next_state, done = memory.sample(batch_size)

    state = torch.tensor(state, dtype=torch.float32)
    next_state = torch.tensor(next_state, dtype=torch.float32)
    action = torch.tensor(action, dtype=torch.long)
    reward = torch.tensor(reward, dtype=torch.float32)
    done = torch.tensor(done, dtype=torch.float32)

    q_values = model(state)
    next_q_values = model(next_state)

    # calculate the q-values of state with the action taken
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    # calculate the q-values of the next state
    next_q_value = torch.max(next_q_values, 1)[0]
    # 0 if next state was 0
    expected_q_value = reward + args.gamma * next_q_value * (1 - done)

    loss = (q_value - expected_q_value.detach()).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def train(iterations: int, batch_size: int):
    state = env.reset()
    losses = []
    all_rewards = []
    episode_reward = 0
    for iteration in range(1, iterations + 1):
        epsilon = epsilon_calculator.value(time_step=iteration)
        action = model.act(state=state, epsilon=epsilon)

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
            loss = compute_td_loss(batch_size=batch_size)
            losses.append(loss.item())

        if iteration % 200 == 0:
            print('Iteration: {0}'.format(iteration))
            print('Rewards: {0}'.format(all_rewards[-9:]))


def play(iterations: int, render=True):
    state = env.reset()
    rewards = []
    episode_reward = 0
    for iteration in range(1, iterations + 1):
        if render:
            env.render()
        action = model.act(state=state, epsilon=0)
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
    parser = argparse.ArgumentParser(description='DQN agent execution')
    parser.add_argument('--env', type=str, metavar='E', default='CartPole-v0', help='GYM environment')
    parser.add_argument('--init_eps', type=float, metavar='I', default=1.0, help='Initial epsilon')
    parser.add_argument('--min_eps', type=float, metavar='M', default=0.01, help='Minimum epsilon')
    parser.add_argument('--eps_decay', type=int, metavar='D', default=500, help='Epsilon decay')
    parser.add_argument('--gamma', type=int, metavar='G', default=0.99, help='Gamma')
    parser.add_argument('--memory_size', type=int, metavar='S', default=10000, help='Memory size')
    parser.add_argument('--batch_size', type=int, metavar='B', default=32, help='Batch size')
    parser.add_argument('--iterations', type=int, metavar='IT', default=10000, help='Training iterations')
    args = parser.parse_args()

    env = gym.make(args.env)
    number_of_observations = env.observation_space.shape[0]
    number_of_actions = env.action_space.n
    model = DQN(num_features=number_of_observations, num_actions=number_of_actions)
    optimizer = optim.Adam(model.parameters())
    memory = ReplayMemory(capacity=args.memory_size)
    epsilon_calculator = ExponentialSchedule(initial_p=args.init_eps, min_p=args.min_eps, decay=args.eps_decay)
    train(iterations=args.iterations, batch_size=args.batch_size)
    play(iterations=10000, render=True)
    torch.save(model.state_dict(), "../../models/dqn.model")
