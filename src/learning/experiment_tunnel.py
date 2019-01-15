import time

import gym
import torch

from src.agents.parallel import ParallelAgent
from src.policy.dqn import DeepQNetwork
from src.policy.dqn_prioritized import PrioritizedDoubleDeepQNetwork
from src.representation.learners import CerberusPixel, Flatten
from src.utils.schedules import LinearSchedule, ExponentialSchedule

if torch.cuda.is_available():
    print("Using GPU - Setting default tensor type to cuda.FloatTensor.")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# MODULES
env = gym.make("Tunnel-v0")
representation_module = CerberusPixel(width=env.observation_space.shape[0],
                                      height=env.observation_space.shape[1],
                                      n_actions=env.action_space.n,
                                      n_hidden=30)
representation_module = Flatten()

memory_delay = 20000
init_eps = 1.0
memory_eps = 0.8
min_eps = 0.01
eps_decay = 20000
linear = LinearSchedule(schedule_timesteps=memory_delay, initial_p=init_eps, final_p=memory_eps)
exponential = ExponentialSchedule(initial_p=memory_eps, min_p=min_eps, decay=eps_decay)
policy = DeepQNetwork(900, env.action_space.n, eps_calculator=linear,
                      memory_eps_calculator=exponential, memory_delay=memory_delay,
                      representation_network=representation_module.network)

# AGENT
agent = ParallelAgent(representation_module, policy, [env])

# CUDA
representation_module.network.to(device)  # if using passthrough or Flatten comment this
policy.model.to(device)
# policy.target_model.to(device)

# TRAIN/TEST
start_time = time.time()
agent.train_agent(100000, log=True, plot_every=1000, episodes_per_saving=10000)
print(f'Total training took {(time.time() - start_time) / 60:.2f} min')
agent.test(numb_runs=20, env=env)
