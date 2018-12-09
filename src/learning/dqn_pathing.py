import argparse

import gym
import torch
from torch import optim
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from src.agents.dqn_agent import DQN
from src.utils.epsilon_calculator import EpsilonCalculator
from src.utils.memory.replay_memory import ReplayMemory
from src.task.pathing import ObstaclePathing


def compute_td_loss(batch_size: int):
    """
    Method that computes the loss of a batch. The batch is sample for memory to take in consideration
    situations that happens before.

    :param model: deep q learning model
    :param optimizer: optimizer used in the process
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
    expected_q_value = reward + model.gamma * next_q_value * (1 - done)

    loss = (q_value - expected_q_value.data).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def train(epochs: int, batch_size: int, max_timesteps: int):
    state = env.reset()
    losses = []
    all_rewards = []
    episode_reward = 0
    for epoch in range(1, epochs + 1):
        epsilon = epsilon_calculator.calculate(iteration=epochs)
        # run an episode
        state = env.reset()
        done = False
        timesteps = 0
        episode_reward = 0
        while not done and timesteps < max_timesteps:
            timesteps += 1
            action = model.act(state=state, epsilon=epsilon)

            next_state, reward, done = env.step(action)
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

        if epoch % 10 == 0:
            print('Iteration: {0}'.format(epoch))
            print('Rewards: {0}'.format(all_rewards[-9:]))


def play(epochs: int, max_timesteps: int, render=True):
    rewards = []
    for epoch in range(1, epochs + 1):
        if render:
            fig = plt.figure(figsize=(10, 6))
        ims = []
        state = env.reset()
        done = False
        episode_reward = 0
        timesteps = 0
        while not done and timesteps < max_timesteps:
            timesteps += 1
            if render:
                im = plt.imshow(env.get_pixelbased_representation(), cmap="binary", origin="upper", animated=True)
                ims.append([im])

            action = model.act(state=state, epsilon=0)
            next_state, reward, done = env.step(action)
            state = next_state
            episode_reward += reward
            if done:
                state = env.reset()
                rewards.append(episode_reward)
                episode_reward = 0
        if not done:
            rewards.append(episode_reward)
        if render:
            ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)
            plt.show()
    # if render:
    #     env.close()
    print('End of game. Final rewards {0}'.format(rewards))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DQN agent execution')
    parser.add_argument('--env', type=str, metavar='E', default='CartPole-v0', help='GYM environment')
    parser.add_argument('--init_eps', type=float, metavar='I', default=1.0, help='Initial epsilon')
    parser.add_argument('--min_eps', type=float, metavar='M', default=0.01, help='Minimum epsilon')
    parser.add_argument('--eps_decay', type=int, metavar='D', default=50, help='Epsilon decay')
    parser.add_argument('--gamma', type=int, metavar='G', default=0.99, help='Gamma')
    parser.add_argument('--memory_size', type=int, metavar='S', default=1000, help='Memory size')
    parser.add_argument('--batch_size', type=int, metavar='B', default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, metavar='EP', default=1000, help='Training epochs')
    parser.add_argument('--max_timesteps', type=int, metavar='TS', default=300, help='Maximal timesteps for epoch')
    args = parser.parse_args()

    size = 30
    env = ObstaclePathing(size, size,
                          [[0, 13, 18, 20],
                           [16, 18, 11, 30],
                           [0, 25, 6, 8]]
                          )
    number_of_observations = 2
    number_of_actions = len(env.action_space)
    model = DQN(num_features=number_of_observations, num_actions=number_of_actions, gamma=args.gamma)
    optimizer = optim.Adam(model.parameters())
    memory = ReplayMemory(capacity=args.memory_size)
    epsilon_calculator = EpsilonCalculator(initial_epsilon=args.init_eps, min_epsilon=args.min_eps,
                                           epsilon_decay=args.eps_decay)

    train_model = False
    if train_model:
        train(epochs=args.epochs, batch_size=args.batch_size, max_timesteps=args.max_timesteps)
        torch.save(model.state_dict(), f"../../models/dqn_pathing_{args.epochs}iter.model")
    else:
        model.load_state_dict(torch.load(f"../../models/dqn_pathing_{args.epochs}iter.model"))
    play(epochs=1, max_timesteps=args.max_timesteps, render=True)

