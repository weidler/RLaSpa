import time

import gym
import numpy as np
import torch

from src.policy.dqn_prioritized import PrioritizedDoubleDeepQNetwork
from src.utils.logger import Logger
from src.utils.schedules import ExponentialSchedule, LinearSchedule


def train(iterations: int):
    state = torch.tensor(env.reset()).float()
    all_rewards = []
    episode_reward = 0
    episode_loss: list = []
    episodes = 0
    steps = 0
    print_ep = 0
    for iteration in range(1, iterations + 1):
        action = model.choose_action(state=state)

        next_state, reward, done, _ = env.step(action)
        next_state = torch.tensor(next_state).float()

        loss = model.update(state, action, reward, next_state, done)

        episode_loss.append(loss)

        state = next_state
        episode_reward += reward
        steps += 1

        if done:
            # model.scheduler.step()
            torch.tensor(env.reset()).float()
            all_rewards.append(episode_reward)
            info = {'loss': np.mean(episode_loss), 'reward': episode_reward, 'mean_reward': np.mean(all_rewards[-10:])}
            logger.scalar_summary_dict(info, iteration)
            episode_reward = 0
            episode_loss.clear()
            episodes += 1

        if episodes % 10 == 0 and print_ep != episodes:
            print_ep = episodes
            print('Iteration: {0}'.format(iteration))
            print('Episode: {0}'.format(episodes))
            print('Rewards: {0}'.format(all_rewards[-10:]))

    model.finish_training()
    print("Played {0} episodes".format(episodes))


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
    env = gym.make('CartPole-v0')
    number_of_observations = env.observation_space.shape[0]
    number_of_actions = env.action_space.n

    memory_delay = 0
    init_eps = 1.0
    memory_eps = 1.0
    min_eps = 0.05
    eps_decay = 10000
    linear = LinearSchedule(schedule_timesteps=memory_delay, initial_p=init_eps, final_p=memory_eps)
    exponential = ExponentialSchedule(initial_p=memory_eps, min_p=min_eps, decay=eps_decay)
    model = PrioritizedDoubleDeepQNetwork(num_features=number_of_observations, num_actions=number_of_actions,
                                          eps_calculator=linear, memory_eps_calculator=exponential,
                                          memory_delay=memory_delay, batch_size=128)

    logger = Logger('logs', 'dqn_playground_{0}'.format(time.time()))

    train(iterations=100000)
    play(iterations=10000, render=True)
