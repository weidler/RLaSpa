import gym
import torch

from src.agents.parallel import ParallelAgent
from src.policy.ddqn import DoubleDeepQNetwork
from src.policy.dqn import DeepQNetwork, PrioritizedDeepQNetwork
from src.representation.learners import PassThrough

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = gym.make("CartPole-v0")
representation_module = PassThrough()
policy = DoubleDeepQNetwork(4, env.action_space.n, eps_decay=10000)

agent = ParallelAgent(representation_module, policy, [env])

# repr_learner.network.to(device)
policy.current_model.to(device)
policy.target_model.to(device)
# policy.model.to(device)

agent.train_agent(50000, log=True, episodes_per_saving=1000)
agent.test(numb_runs=10, env=env, visual=False)
