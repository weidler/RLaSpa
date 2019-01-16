import time

import gym
import torch
import src.gym_custom_tasks
from src.agents.parallel import ParallelAgent
from src.policy.dqn import DoubleDeepQNetwork
from src.representation.learners import JanusPixel, CerberusPixel, CVAEPixel
from src.utils.schedules import LinearSchedule, ExponentialSchedule
import sys

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("Using GPU - Setting default tensor type to cuda.FloatTensor.")
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # MODULES
    # ENV
    envs_name = sys.argv[0].lower()
    if envs_name == 'tunnel':
        envs = [gym.make("Tunnel-v0")]
    elif envs_name == 'scrollers':
        envs = [gym.make("Tunnel-v0"),
                gym.make("Evasion-v0"),
                gym.make("EvasionWalls-v0")]
    elif envs_name == 'pathing':
        envs = [gym.make("VisualObstaclePathing-v0"),
                gym.make("VisualObstaclePathing-v1"),
                gym.make("VisualObstaclePathing-v2"),
                gym.make("VisualObstaclePathing-v3")
                ]
    else:
        raise ValueError('No such env {}'.format(envs_name))

    # REPRESENTATION
    repr_learner_name = sys.argv[1].lower()
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
    else:
        raise ValueError('No such repr learner {}'.format(repr_learner_name))

    memory_delay = 100000
    init_eps = 1.0
    memory_eps = 0.8
    min_eps = 0.01
    eps_decay = 40000000
    episode = 1000000
    linear = LinearSchedule(schedule_timesteps=memory_delay, initial_p=init_eps, final_p=memory_eps)
    exponential = ExponentialSchedule(initial_p=memory_eps, min_p=min_eps, decay=eps_decay)
    policy = DoubleDeepQNetwork(representation_module.n_hidden, envs[0].action_space.n, eps_calculator=linear,
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
    agent.train_agent(episode, log=True, episodes_per_saving=10000)
    agent.save(episode=episode)  # save last
    print(f'Total training took {(time.time() - start_time) / 60:.2f} min')
    for env in envs:
        agent.test(numb_runs=100, env=env)
