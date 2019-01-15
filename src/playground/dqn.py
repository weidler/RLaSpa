import time

import gym
import torch

from src.policy.dqn import DoubleDeepQNetwork
from src.utils.logger import Logger
from src.utils.schedules import ExponentialSchedule, LinearSchedule


def train(iterations: int):
    state = torch.tensor(env.reset()).float()
    losses = []
    all_rewards = []
    episode_reward = 0
    episode_loss = 0
    episodes = 0
    for iteration in range(1, iterations + 1):
        action = model.choose_action(state=state)

        next_state, reward, done, _ = env.step(action)
        next_state = torch.tensor(next_state).float()

        loss = model.update(state, action, reward, next_state, done)

        episode_loss += loss

        state = next_state
        episode_reward += reward

        if done:
            info = {'loss': episode_loss, 'reward': episode_reward}
            logger.scalar_summary_dict(info, iteration)
            torch.tensor(env.reset()).float()
            all_rewards.append(episode_reward)
            losses.append(episode_loss)
            episode_reward = 0
            episode_loss = 0
            episodes += 1

        if iteration % 200 == 0:
            print('Iteration: {0}'.format(iteration))
            print('Rewards: {0}'.format(all_rewards[-9:]))
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
    memory_eps = 0.8
    min_eps = 0.001
    eps_decay = 10000
    linear = LinearSchedule(schedule_timesteps=memory_delay, initial_p=init_eps, final_p=memory_eps)
    exponential = ExponentialSchedule(initial_p=memory_eps, min_p=min_eps, decay=eps_decay)
    model = DoubleDeepQNetwork(num_features=number_of_observations, num_actions=number_of_actions,
                               eps_calculator=linear, memory_eps_calculator=exponential,
                               memory_delay=memory_delay)

    logger = Logger('logs', 'dqn_playground_{0}'.format(time.time()))

    train(iterations=50000)
    play(iterations=10000, render=True)
