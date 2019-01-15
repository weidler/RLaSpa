import gym
import torch

from src.agents.parallel import ParallelAgent
from src.policy.dqn_prioritized import PrioritizedDoubleDeepQNetwork
from src.representation.learners import Flatten
from src.utils.schedules import LinearSchedule, ExponentialSchedule

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = gym.make("Race-v0")
representation_module = Flatten()
memory_delay = 5000
init_eps = 1.0
memory_eps = 0.8
min_eps = 0.01
eps_decay = 2000
linear = LinearSchedule(schedule_timesteps=memory_delay, initial_p=init_eps, final_p=memory_eps)
exponential = ExponentialSchedule(initial_p=memory_eps, min_p=min_eps, decay=eps_decay)
policy = PrioritizedDoubleDeepQNetwork(900, env.action_space.n, eps_calculator=linear,
                                       memory_eps_calculator=exponential, memory_delay=memory_delay)
agent = ParallelAgent(representation_module, policy, [env])

# repr_learner.network.to(device)
policy.model.to(device)
policy.target_model.to(device)


agent.train_agent(100000, log=True)
agent.test(numb_runs=20, env=env)
