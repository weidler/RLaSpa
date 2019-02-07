import time

import gym
from gym import Env
import torch
import src.gym_custom_tasks
from src.agents.parallel import ParallelAgent
from src.policy.dqn import DoubleDeepQNetwork
from src.representation.learners import JanusPixel, CerberusPixel, CVAEPixel, ConvolutionalPixel
from src.utils.schedules import LinearSchedule, ExponentialSchedule
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def create_baseline(env: Env, numb_runs: int = 1, render: bool = False, visual=True) -> None:
    """
    Run a test in the environment using the current policy without exploration.

    :param numb_runs: number of test to be done.
    :param render: render the environment
    """

    all_rewards = []
    fig = plt.figure(figsize=(10, 6))
    for i in range(numb_runs):
        plt.clf()
        ims = []
        done = False
        env.reset()
        step = 0
        total_reward = 0
        while not done:
            action = np.random.choice(env.action_space.n+1)
            state, reward, done, _ = env.step(action)
            step += 1
            total_reward += reward
            if visual: ims.append([plt.imshow(state.cpu(), cmap="binary", origin="upper", animated=True)])
            if render:
                env.render()
        if not render and visual:
            ani = animation.ArtistAnimation(fig, ims, blit=True, repeat_delay=1000)
            ani.save((f'../../../data/{env.__class__.__name__}_testrun_{i}.gif'), writer="imagemagick", fps=15)
        all_rewards.append(total_reward)
        print(f"Tested episode {i} took {step} steps and gathered a reward of {total_reward}.")
    print(f'Average max score after {numb_runs} testruns: {sum(all_rewards) / len(all_rewards)} with a peak of {max(all_rewards)} at episode {all_rewards.index(max(all_rewards))}')


if __name__ == '__main__':
    if torch.cuda.is_available():
        print("Using GPU - Setting default tensor type to cuda.FloatTensor.")
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ### Environment to test on
    envs = [gym.make('Evasion-v0')]
    ### PATH to checkpoint of model that was trained on tasks
    # trained_model = '../../../ckpt/ParallelAgent_Tunnel-v0_JanusPixel_DoubleDeepQNetwork/milestone_500000'
    trained_model = '../../../ckpt/ParallelAgent_Tunnel-v0_Evasion-v0_EvasionWalls-v0_CerberusPixel_DoubleDeepQNetwork/2019-01-21_02-04-24'
    # MODULES
    # REPRESENTATION (Needs to be the same that it is trained on)
    repr_learner_name = 'cerberus'

    if repr_learner_name == 'janus':
        representation_module = JanusPixel(width=envs[0].observation_space.shape[0],
                                           height=envs[0].observation_space.shape[1],
                                           n_actions=envs[0].action_space.n,
                                           n_hidden=32)

    elif repr_learner_name == 'cerberus':
        representation_module = CerberusPixel(width=envs[0].observation_space.shape[0],
                                              height=envs[0].observation_space.shape[1],
                                              n_actions=envs[0].action_space.n,
                                              n_hidden=32)
    elif repr_learner_name == 'cvae':
        representation_module = CVAEPixel(n_middle=32, n_hidden=16)
    elif repr_learner_name == 'cae':
        representation_module = ConvolutionalPixel(n_output=32)
    else:
        raise ValueError('No such repr learner {}'.format(repr_learner_name))

    memory_delay = 100000
    init_eps = 1.0
    memory_eps = 0.8
    min_eps = 0.01
    eps_decay = 3000000
    episode = 1000000
    linear = LinearSchedule(schedule_timesteps=memory_delay, initial_p=init_eps, final_p=memory_eps)
    exponential = ExponentialSchedule(initial_p=memory_eps, min_p=min_eps, decay=eps_decay)
    policy = DoubleDeepQNetwork(32, envs[0].action_space.n, eps_calculator=linear,
                                memory_eps_calculator=exponential, memory_delay=memory_delay,
                                representation_network=representation_module.network)

    # AGENT
    agent = ParallelAgent(representation_module, policy, envs)

    # CUDA
    representation_module.network.to(device)  # if using passthrough or Flatten comment this
    policy.model.to(device)
    policy.target_model.to(device)

    # Create baselines
    create_baseline(envs[0], 1000, render=False, visual=False)

    # TRAIN/TEST
    # start_time = time.time()
    # agent.load(trained_model, gpu=torch.cuda.is_available())
    # print(f'Total training took {(time.time() - start_time) / 60:.2f} min')
    # for env in envs:
    #     agent.test(numb_runs=1000, env=env, render=True)
