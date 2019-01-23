import time

import gym
import torch
import src.gym_custom_tasks
from src.agents.parallel import ParallelAgent
from src.policy.dqn import DoubleDeepQNetwork
from src.representation.learners import JanusPixel, CerberusPixel, CVAEPixel, ConvolutionalPixel
from src.utils.schedules import LinearSchedule, ExponentialSchedule
import sys

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("Using GPU - Setting default tensor type to cuda.FloatTensor.")
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    envs = [gym.make("Race-v0")]



    representation_module = JanusPixel(width=envs[0].observation_space.shape[0],
                                       height=envs[0].observation_space.shape[1],
                                       n_actions=envs[0].action_space.n,
                                       n_hidden=32)

    memory_delay = 100000
    init_eps = 1.0
    memory_eps = 0.8
    min_eps = 0.01
    eps_decay = 300000
    episode = 1050000
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

    # TRAIN/TEST
    start_time = time.time()
    agent.load("/home/alescop/Desktop/final_experiments/zero-shot/janus_tunnel", load_policy_learner=True)
    agent.train_agent(episode, train_on_ae_loss=False, plot_every=25000, episodes_per_saving=2000, statistics_every=500, stat_label="1m_tunnel")
    print(f'Total training took {(time.time() - start_time) / 60:.2f} min')
    for env in envs:
        agent.test(numb_runs=10000, env=env)

