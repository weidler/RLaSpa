import torch

import gym

from src.agents.parallel import ParallelAgent
from src.policy.ddqn import DoubleDeepQNetwork
from src.policy.tablebased import QTableOffPolicy
from src.representation.learners import Flatten, PassThrough

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = gym.make("SimplePathing-v1")
representation_module = PassThrough()
policy = QTableOffPolicy([10, 10], 4, temperature=3)

agent = ParallelAgent(representation_module, policy, [env])

# repr_learner.network.to(device)
# policy.current_model.to(device)
# policy.target_model.to(device)

agent.train_agent(10000, log=True)
agent.test(numb_runs=20, env=env, visual=False)