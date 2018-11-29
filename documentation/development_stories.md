# Development histories
The function of this document is to write all the things that can be useful for the rest.

## GYM
The environments informations are:

|Environment Id|Observation Space|Action Space|Reward Range|tStepL|Trials|rThresh|
|---|---|---|---|---|---|---|
| [MountainCar-v0](https://gym.openai.com/envs/MountainCar-v0)|Box(2,)|Discrete(3)|(-inf, inf)|200|100|-110.0|
| [Pendulum-v0](https://gym.openai.com/envs/Pendulum-v0)|Box(3,)|Box(1,)|(-inf, inf)|200|100|None|
| [CartPole-v0](https://gym.openai.com/envs/CartPole-v0)|Box(4,)|Discrete(2)|(-inf, inf)|200|100|195.0|
| [Acrobot-v0](https://gym.openai.com/envs/Acrobot-v0)|Box(4,)|Discrete(3)|(-inf, inf)|200|100|-100|

- [Cartpole](https://github.com/openai/gym/wiki/CartPole-v0)
- [Pendulum](https://github.com/openai/gym/wiki/Pendulum-v0)
- [Mountain Car](https://github.com/openai/gym/wiki/MountainCar-v0)
- [Acrobot](https://github.com/openai/gym/blob/master/gym/envs/classic_control/acrobot.py)
## Pytorch

### GPU optimization

- PyTorch needs an tensor of N x num_features, where N is the batch size. Unsqueeze, effectively, 
  turns `[0,1,2]` into `[[0,1,2]]` that is, a batch size of one, one action predicted, given the current
  state, at a time. Conceptually simple but probably something you’d want to optimize for production, 
  otherwise GPU capacity is being wasted. 