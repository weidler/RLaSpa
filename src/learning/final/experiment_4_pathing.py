import torch

import gym

from src.agents.parallel import ParallelAgent
from src.policy.dqn import DoubleDeepQNetwork
from src.representation.learners import JanusPixel, CerberusPixel, CVAEPixel
from src.utils.schedules import LinearSchedule, ExponentialSchedule

import sys

if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    environments = [
        gym.make("VisualObstaclePathing-v0"),
        gym.make("VisualObstaclePathing-v1"),
        gym.make("VisualObstaclePathing-v2"),
        gym.make("VisualObstaclePathing-v3")
    ]

    # REPRESENTATION
    repr_learner_name = sys.argv[0].lower()

    if repr_learner_name == 'janus':
        repr_learner = JanusPixel(width=environments[0].observation_space.shape[0],
                                  height=environments[0].observation_space.shape[1],
                                  n_actions=environments[0].action_space.n,
                                  n_hidden=32)
    elif repr_learner_name == 'cerberus':
        repr_learner = CerberusPixel(width=environments[0].observation_space.shape[0],
                                     height=environments[0].observation_space.shape[1],
                                     n_actions=environments[0].action_space.n,
                                     n_hidden=32)
    elif repr_learner_name == 'cvae':
        repr_learner = CVAEPixel(n_middle=64, n_hidden=32)
    else:
        raise ValueError('No such repr learner {}'.format(repr_learner_name))

    episode = 1000000
    memory_delay = 100000
    init_eps = 1.0
    memory_eps = 0.8
    min_eps = 0.01
    eps_decay = 40000000

    linear = LinearSchedule(schedule_timesteps=memory_delay, initial_p=init_eps, final_p=memory_eps)
    exponential = ExponentialSchedule(initial_p=memory_eps, min_p=min_eps, decay=eps_decay)
    policy = DoubleDeepQNetwork(repr_learner.n_hidden, environments[0].action_space.n, eps_calculator=linear,
                                memory_eps_calculator=exponential, memory_delay=memory_delay, representation_network=repr_learner.network)

    agent = ParallelAgent(repr_learner, policy, environments)

    repr_learner.network.to(device)
    policy.model.to(device)
    policy.target_model.to(device)

    agent.train_agent(episodes=episode, log=True, episodes_per_saving=10000)
    agent.save(episode=episode)  # save last

    # print(f'Total training took {(time.time() - start_time) / 60:.2f} min')
    for env in environments:
        agent.test(numb_runs=100, env=env)
