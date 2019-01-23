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

    # MODULES
    # ENV
    # envs_name = sys.argv[1].lower()
    # if envs_name == 'tunnel':
    #     envs = [gym.make("Tunnel-v0")]
    # elif envs_name == 'race':
    #     envs = [gym.make("Race-v0")]
    # elif envs_name == 'scrollers':
    #     envs = [gym.make("Tunnel-v0"),
    #             gym.make("Evasion-v0"),
    #             gym.make("EvasionWalls-v0")]
    # elif envs_name == 'pathing':
    #     envs = [gym.make("VisualObstaclePathing-v0"),
    #             gym.make("VisualObstaclePathing-v1"),
    #             gym.make("VisualObstaclePathing-v2"),
    #             gym.make("VisualObstaclePathing-v3")
    #             ]
    # else:
    #     raise ValueError('No such env {}'.format(envs_name))
    #
    # # REPRESENTATION
    # repr_learner_name = sys.argv[2].lower()
    # if repr_learner_name == 'janus':
    #     representation_module = JanusPixel(width=envs[0].observation_space.shape[0],
    #                                        height=envs[0].observation_space.shape[1],
    #                                        n_actions=envs[0].action_space.n,
    #                                        n_hidden=32)
    #
    # elif repr_learner_name == 'cerberus':
    #     representation_module = CerberusPixel(width=envs[0].observation_space.shape[0],
    #                                           height=envs[0].observation_space.shape[1],
    #                                           n_actions=envs[0].action_space.n,
    #                                           n_hidden=32)
    # elif repr_learner_name == 'cvae':
    #     representation_module = CVAEPixel(n_middle=32, n_hidden=16)
    # elif repr_learner_name == 'cae':
    #     representation_module = ConvolutionalPixel(n_output=32)
    # else:
    #     raise ValueError('No such repr learner {}'.format(repr_learner_name))


    envs = [gym.make("Race-v0")]



    representation_module = JanusPixel(width=envs[0].observation_space.shape[0],
                                       height=envs[0].observation_space.shape[1],
                                       n_actions=envs[0].action_space.n,
                                       n_hidden=32)

    memory_delay = 100000
    init_eps = 1.0
    memory_eps = 0.8
    min_eps = 0.01
    eps_decay = 40000000
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

    # TRAIN/TEST
    start_time = time.time()
    # agent.load("/home/alescop/Desktop/final_experiments/zero-shot/janus_scroll", load_policy_learner=True)
    agent.train_agent(10000, plot_every=9999)
    print(f'Total training took {(time.time() - start_time) / 60:.2f} min')
    for env in envs:
        agent.test(numb_runs=10000, env=env)

