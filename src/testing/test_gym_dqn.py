import argparse

import gym
import torch
from torch import optim

from src.dqn.dqn import DQN


def compute_td_loss(model: DQN, optimizer, batch_size):
    state, action, reward, next_state, done = model.replay_memory.sample(batch_size)

    state = torch.tensor(state, dtype=torch.float32)
    next_state = torch.tensor(next_state, dtype=torch.float32)
    action = torch.tensor(action, dtype=torch.long)
    reward = torch.tensor(reward, dtype=torch.float32)
    done = torch.tensor(done, dtype=torch.float32)

    q_values = model(state)
    next_q_values = model(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + model.gamma * next_q_value * (1 - done)

    loss = (q_value - expected_q_value.data).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def train(env, model: DQN, optimizer, iterations: int, batch_size: int):
    state = env.reset()
    losses = []
    all_rewards = []
    episode_reward = 0
    for iteration in range(1, iterations + 1):
        epsilon = model.calculate_epsilon(iteration)
        action = model.act(state, epsilon)

        next_state, reward, done, _ = env.step(action)
        model.replay_memory.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if done:
            state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0

        if len(model.replay_memory) > batch_size:
            loss = compute_td_loss(model=model, optimizer=optimizer, batch_size=batch_size)
            losses.append(loss.data[0])

        if iteration % 200 == 0:
            print('Iteration: {0}'.format(iteration))
            print('Rewards: {0}'.format(all_rewards[-9:]))


def main(arguments):
    env = gym.make(arguments.env)
    number_of_observations = env.observation_space.shape[0]
    number_of_actions = env.action_space.n
    model = DQN(num_features=number_of_observations, num_actions=number_of_actions, init_epsilon=arguments.init_eps,
                min_epsilon=arguments.min_eps, epsilon_decay=arguments.eps_decay, gamma=args.gamma,
                memory_size=arguments.memory_size)
    # check cuda compatibility
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = optim.Adam(model.parameters())
    train(env, model, optimizer=optimizer, iterations=args.iterations, batch_size=args.batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DQN agent execution')
    parser.add_argument('--env', type=str, metavar='E', nargs=1, default='CartPole-v0', help='GYM environment')
    parser.add_argument('--init_eps', type=float, metavar='I', nargs=1, default=1.0, help='Initial epsilon')
    parser.add_argument('--min_eps', type=float, metavar='M', nargs=1, default=0.01, help='Minimum epsilon')
    parser.add_argument('--eps_decay', type=int, metavar='D', nargs=1, default=500, help='Epsilon decay')
    parser.add_argument('--gamma', type=int, metavar='G', nargs=1, default=0.9, help='Gamma')
    parser.add_argument('--memory_size', type=int, metavar='S', nargs=1, default=10000, help='Memory size')
    parser.add_argument('--batch_size', type=int, metavar='B', nargs=1, default=32, help='Batch size')
    parser.add_argument('--iterations', type=int, metavar='IT', nargs=1, default=10000, help='Training iterations')
    args = parser.parse_args()
    main(args)
