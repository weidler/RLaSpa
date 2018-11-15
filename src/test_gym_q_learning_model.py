import gym
import numpy as np
import torch
from representation.simple_autoencoder import Autoencoder
from sklearn.utils.extmath import cartesian


# env = gym.make("Taxi-v2")
env = gym.make("CartPole-v1")
env.reset()  # returns initial state

# We can determine the total number of possible states using the following command:
# env.observation_space.n
# If you would like to visualize the current state, type the following:
# env.render()
# let's explore the actions available to the agent.
# env.action_space.n


def binThese(lowerBound, upperBound, bins=10, cutOffInf=10000):
    assert len(upperBound) == len(lowerBound)
    numFeatures = len(upperBound)
    featureBins = np.zeros((numFeatures, bins+1))
    for i in range(numFeatures):
        lb = -cutOffInf if lowerBound[i] <= -cutOffInf else lowerBound[i]
        ub = cutOffInf if upperBound[i] >= cutOffInf else upperBound[i]
        featureBins[i] = np.histogram_bin_edges([lb, ub], bins=bins)
    return featureBins[:, :-1]


def binModel(lowerBound, upperBound, numFeatures, bins=10):
    featureBins = np.zeros((numFeatures, bins+1))
    for i in range(numFeatures):
        lb = lowerBound
        ub = upperBound
        featureBins[i] = np.histogram_bin_edges([lb, ub], bins=bins)
    return featureBins[:, :-1]


def getState(allStates, stateBins, features):
    # print(len(stateBins))
    # print(len(features))
    assert len(stateBins) == len(features)
    bin = [np.where(stateBins[i] <= f)[0][-1] for i, f in enumerate(features)]
    # stateIndex = feature1*binSize^numRestFeatures + feature2*binSize^numRestFeatures...
    stateIndex = int(np.sum(bin * np.power(len(stateBins[0]), np.linspace(len(features)-1, 0, len(features)))))
    return stateIndex


net = Autoencoder()
net.load_state_dict(torch.load('../models/very-simple.model'))

numFeatures = net.encoder.out_features

# in CartPole-v0, parameters are:
# [position of cart, velocity of cart, angle of pole, rotation rate of pole]
# print('low', env.observation_space.low)
# print('high', env.observation_space.high)
bins = 50
# stateBins = binThese(env.observation_space.low, env.observation_space.high, bins=bins, cutOffInf=5)
stateBins = binModel(0, 1, numFeatures, bins=bins)
allStates = cartesian(stateBins)  # round

# print(stateBins)
print(allStates[0, :])

qValues = np.zeros([len(allStates), env.action_space.n])
totalReward = 0
alpha = 0.618
total_episode = 10000
print(f'qValues.shape: {qValues.shape}')

for episode in range(1, total_episode):
    done = False
    totalReward, reward = 0, 0
    state = env.reset()
    explore_rate = (1 - episode / total_episode) / 10 + 0.05
    # here we need to find the state of all states
    # transform to encoder latent representation
    state = net.activation(net.encoder(torch.tensor(state).float())).tolist()
    # print(f'latent: {state}')
    stateInd = getState(allStates, stateBins, state)
    while not done:
            if np.random.rand() > explore_rate:  # exploit
                action = np.argmax(qValues[stateInd])  # 1
            else:  # explore
                action = np.random.randint(0, len(qValues[0]))
            # print(f'action: {action}')
            internalRepr, reward, done, info = env.step(action)  # 2
            # print(reward)
            latentRepr = net.activation(net.encoder(torch.tensor(internalRepr).float())).tolist()
            # print(f'asdawdwdad {latentRepr}')
            indState2 = getState(allStates, stateBins, latentRepr)  # find the state bin
            # indState2 = np.where(allStates == state2)[0][0]
            qValues[stateInd, action] += alpha * (reward + np.max(qValues[indState2]) - qValues[stateInd, action])  # 3
            totalReward += reward
            stateInd = indState2
            # print(stateInd)
    if episode % 50 == 0:
        print('Episode {}, explore rate {}, Total Reward: {}'.format(episode, round(explore_rate, 2), totalReward))

print(np.sum(qValues)/np.count_nonzero(qValues))

done = False
state = env.reset()
test_total_reward = 0
while not done:
    env.render()
    state = net.activation(net.encoder(torch.tensor(state).float())).tolist()
    stateInd = getState(allStates, stateBins, state)
    # stateInd = np.where(allStates == state)[0][0]
    action = np.argmax(qValues[stateInd])
    state, reward, done, _ = env.step(action)
    test_total_reward += reward
print("Ended with reward: ", test_total_reward)

env.close()
