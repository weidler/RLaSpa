import gym
import torch

from src.agents.parallel import ParallelAgent
from src.policy.dqn import DeepQNetwork
from src.representation.learners import Flatten
from src.utils.schedules import LinearSchedule, ExponentialSchedule

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = gym.make("Evasion-v0")
representation_module = Flatten()
memory_delay = 5000
init_eps = 1.0
memory_eps = 0.8
min_eps = 0.01
eps_decay = 50000
linear = LinearSchedule(schedule_timesteps=memory_delay, initial_p=init_eps, final_p=memory_eps)
exponential = ExponentialSchedule(initial_p=memory_eps, min_p=min_eps, decay=eps_decay)
policy = DeepQNetwork(900, env.action_space.n, eps_calculator=linear,
                      memory_eps_calculator=exponential, memory_delay=memory_delay)

agent = ParallelAgent(representation_module, policy, env)

policy.model.to(device)

agent.train_agent(10000, log=True, episodes_per_saving=1000)
agent.test(numb_runs=10, env=env)
