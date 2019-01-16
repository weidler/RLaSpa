import time

import gym
import torch

from src.agents.parallel import ParallelAgent
from src.policy.dqn import DeepQNetwork, DoubleDeepQNetwork
from src.policy.dqn_prioritized import PrioritizedDoubleDeepQNetwork
from src.representation.learners import CerberusPixel, Flatten, ConvolutionalPixel
from src.representation.visual.pixelencoder import ConvolutionalNetwork
from src.utils.schedules import LinearSchedule, ExponentialSchedule

if torch.cuda.is_available():
    print("Using GPU - Setting default tensor type to cuda.FloatTensor.")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# MODULES
env = gym.make("Tunnel-v0")
env2 = gym.make("EvasionWalls-v0")

representation_module = ConvolutionalPixel(32)

memory_delay = 20000
init_eps = 1.0
memory_eps = 0.8
min_eps = 0.01
eps_decay = 20000
linear = LinearSchedule(schedule_timesteps=memory_delay, initial_p=init_eps, final_p=memory_eps)
exponential = ExponentialSchedule(initial_p=memory_eps, min_p=min_eps, decay=eps_decay)
policy = DoubleDeepQNetwork(32, env.action_space.n, eps_calculator=linear,
                      memory_eps_calculator=exponential, memory_delay=memory_delay,
                      representation_network=representation_module.network)

# AGENT
agent = ParallelAgent(representation_module, policy, [env])

# CUDA
representation_module.network.to(device)  # if using passthrough or Flatten comment this
policy.model.to(device)
policy.target_model.to(device)

# TRAIN/TEST
start_time = time.time()
# agent.load('../../ckpt/ParallelAgent_Tunnel-v0_ConvolutionalPixel_DoubleDeepQNetwork/2019-01-16_01-18-19')
agent.train_agent(20000, plot_every=500, log=True, episodes_per_saving=1000)
print(f'Total training took {(time.time() - start_time) / 60:.2f} min')
agent.test(numb_runs=100, env=env)
