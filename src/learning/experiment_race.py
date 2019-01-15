import torch

import gym

from src.agents.parallel import ParallelAgent
from src.policy.ddqn import DoubleDeepQNetwork
from src.policy.dqn import DeepQNetwork
from src.representation.learners import Flatten, JanusPixel
from src.representation.visual.pixelencoder import JanusPixelEncoder

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = gym.make("Race-v0")
representation_module = JanusPixel(30, 30, 3, 30)
policy = DeepQNetwork(30, env.action_space.n, eps_decay=2000, learning_rate=5e-4)

agent = ParallelAgent(representation_module, policy, [env])

representation_module.network.to(device)
# policy.current_model.to(device)
policy.model.to(device)

agent.train_agent(1000, log=True)
agent.test(numb_runs=20, env=env)