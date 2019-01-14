import time

import gym
import torch

from src.agents.parallel import ParallelAgent
from src.policy.ddqn import DoubleDeepQNetwork
from src.representation.learners import CerberusPixel, Flatten

if torch.cuda.is_available():
    print("Using GPU - Setting default tensor type to cuda.FloatTensor.")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# MODULES
env = gym.make("Tunnel-v0")
# representation_module = CerberusPixel(width=env.observation_space.shape[0],
#                                       height=env.observation_space.shape[1],
#                                       n_actions=env.action_space.n,
#                                       n_hidden=30)
representation_module = Flatten()
policy = DoubleDeepQNetwork(900, env.action_space.n, eps_decay=10000)

# AGENT
agent = ParallelAgent(representation_module, policy, [env])

# CUDA
# representation_module.network.to(device)  # if using passthrough or Flatten comment this
policy.current_model.to(device)
policy.target_model.to(device)

# TRAIN/TEST
start_time = time.time()
agent.train_agent(100, log=True)
print(f'Total training took {(time.time()-start_time)/60:.2f} min')
agent.test(numb_runs=10, env=env)
