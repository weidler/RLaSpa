import argparse

import gym
import torch

from src.policy.dqn_prioritized import PrioritizedDuelingDeepQNetwork
from src.utils.schedules import ExponentialSchedule, LinearSchedule


def train(iterations: int):
    state = torch.tensor(env.reset()).float()
    losses = []
    all_rewards = []
    episode_reward = 0
    episode_loss = 0
    for iteration in range(1, iterations + 1):
        action = model.choose_action(state=state)

        next_state, reward, done, _ = env.step(action)
        next_state = torch.tensor(next_state).float()

        episode_loss += model.update(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if done:
            torch.tensor(env.reset()).float()
            all_rewards.append(episode_reward)
            losses.append(episode_loss)
            episode_reward = 0
            episode_loss = 0

        if iteration % 200 == 0:
            print('Iteration: {0}'.format(iteration))
            print('Rewards: {0}'.format(all_rewards[-9:]))


def play(iterations: int, render=True):
    state = torch.tensor(env.reset()).float()
    rewards = []
    episode_reward = 0
    for iteration in range(1, iterations + 1):
        if render:
            env.render()
        action = model.choose_action_policy(state=state)
        next_state, reward, done, _ = env.step(action)
        next_state = torch.tensor(next_state).float()
        episode_reward += reward
        state = next_state
        if done:
            state = torch.tensor(env.reset()).float()
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
    parser.add_argument('--iterations', type=int, metavar='IT', default=30000, help='Training iterations')
    args = parser.parse_args()

    env = gym.make(args.env)
    number_of_observations = env.observation_space.shape[0]
    number_of_actions = env.action_space.n
    memory_delay = 0
    init_eps = 1.0
    memory_eps = 0.8
    min_eps = 0.01
    eps_decay = 10000
    linear = LinearSchedule(schedule_timesteps=memory_delay, initial_p=init_eps, final_p=memory_eps)
    exponential = ExponentialSchedule(initial_p=memory_eps, min_p=min_eps, decay=eps_decay)
    model = PrioritizedDuelingDeepQNetwork(num_features=number_of_observations, num_actions=number_of_actions,
                                           eps_calculator=linear, memory_eps_calculator=exponential,
                                           memory_delay=memory_delay)
    train(iterations=args.iterations)
    play(iterations=10000, render=True)
