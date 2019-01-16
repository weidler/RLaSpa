import time

import torch

import gym

from src.agents.parallel import ParallelAgent
from src.policy.dqn import DoubleDeepQNetwork
from src.representation.learners import CerberusPixel, Flatten, ConvolutionalPixel
from src.utils.schedules import LinearSchedule, ExponentialSchedule

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = gym.make("EvasionWalls-v0")

# representation_module = CerberusPixel(width=env.observation_space.shape[0],
#                                  height=env.observation_space.shape[1],
#                                  n_actions=env.action_space.n,
#                                  n_hidden=32)
# representation_module = Flatten()

representation_module = ConvolutionalPixel(512)


memory_delay = 10000
init_eps = 1.0
memory_eps = 0.8
min_eps = 0.01
eps_decay = 10000
linear = LinearSchedule(schedule_timesteps=memory_delay, initial_p=init_eps, final_p=memory_eps)
exponential = ExponentialSchedule(initial_p=memory_eps, min_p=min_eps, decay=eps_decay)
policy = DoubleDeepQNetwork(representation_module.n_hidden, env.action_space.n, eps_calculator=linear,
                            memory_eps_calculator=exponential, memory_delay=memory_delay, representation_network=representation_module.network)

agent = ParallelAgent(representation_module, policy, [env])

representation_module.network.to(device)  # if using passthrough or Flatten comment this
policy.model.to(device)
policy.target_model.to(device)

start_time = time.time()
agent.train_agent(10000, log=True, episodes_per_saving=1000, plot_every=100)
print(f'Total training took {(time.time()-start_time)/60:.2f} min')
agent.test(numb_runs=10, env=env)
