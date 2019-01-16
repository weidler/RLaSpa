import torch

import gym

from src.agents.parallel import ParallelAgent
from src.policy.dqn import DoubleDeepQNetwork
from src.representation.learners import Flatten, ConvolutionalPixel
from src.utils.schedules import LinearSchedule, ExponentialSchedule

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = gym.make("VisualObstaclePathing-v0")
# representation_module = Flatten()
repr_learner = ConvolutionalPixel(512)

memory_delay = 10000
init_eps = 1.0
memory_eps = 0.8
min_eps = 0.01
eps_decay = 10000
linear = LinearSchedule(schedule_timesteps=memory_delay, initial_p=init_eps, final_p=memory_eps)
exponential = ExponentialSchedule(initial_p=memory_eps, min_p=min_eps, decay=eps_decay)
policy = DoubleDeepQNetwork(512, env.action_space.n, eps_calculator=linear,
                            memory_eps_calculator=exponential, memory_delay=memory_delay, representation_network=repr_learner.network)

policy = DoubleDeepQNetwork(512, env.action_space.n, eps_calculator=linear,
                            memory_eps_calculator=exponential, memory_delay=memory_delay, representation_network=repr_learner.network)

agent = ParallelAgent(repr_learner, policy, [env])

repr_learner.network.to(device)
policy.model.to(device)
policy.target_model.to(device)

agent.train_agent(1000, log=True, plot_every=20)
agent.test(numb_runs=20, env=env)
