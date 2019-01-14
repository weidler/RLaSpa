import time

import torch

import gym

from src.agents.parallel import ParallelAgent
from src.policy.ddqn import DoubleDeepQNetwork
from src.policy.dqn import DeepQNetwork
from src.representation.learners import Flatten, CerberusPixel

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = gym.make("EvasionWalls-v0")
representation_module = CerberusPixel(width=env.observation_space.shape[0],
                                 height=env.observation_space.shape[1],
                                 n_actions=env.action_space.n,
                                 n_hidden=30)
policy = DoubleDeepQNetwork(30, env.action_space.n, eps_decay=10000)

agent = ParallelAgent(representation_module, policy, env)

representation_module.network.to(device)  # if using passthrough or Flatten comment this
policy.current_model.to(device)
policy.target_model.to(device)

start_time = time.time()
agent.train_agent(10000, log=True, episodes_per_saving=1000)
print(f'Total training took {(time.time()-start_time)/60:.2f} min')
agent.test(numb_runs=10, env=env)