import torch

import gym

from src.agents.parallel import ParallelAgent
from src.policy.ddqn import DoubleDeepQNetwork
from src.policy.dqn import DeepQNetwork
from src.representation.learners import Flatten

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = gym.make("Evasion-v0")
representation_module = Flatten()
policy = DeepQNetwork(900, env.action_space.n, eps_decay=50000)

agent = ParallelAgent(representation_module, policy, env)

policy.model.to(device)

agent.train_agent(10000, log=True, episodes_per_saving=1000)
agent.test(numb_runs=10)