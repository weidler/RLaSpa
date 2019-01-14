import torch

import gym

from src.agents.parallel import ParallelAgent
from src.policy.ddqn import DoubleDeepQNetwork
from src.representation.learners import Flatten

if torch.cuda.is_available():
    print("Using GPU - Setting default tensor type to cuda.FloatTensor.")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = gym.make("Race-v0")
representation_module = Flatten()
policy = DoubleDeepQNetwork(900, env.action_space.n, eps_decay=2000, learning_rate=5e-4)

agent = ParallelAgent(representation_module, policy, env)

# repr_learner.network.to(device)
policy.current_model.to(device)
policy.target_model.to(device)

agent.train_agent(40000, log=True, episodes_per_saving=10000)
agent.test(runs_number=20)