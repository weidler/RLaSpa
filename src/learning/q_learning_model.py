import gym
import numpy as np
import torch
from src.representation.autoencoder import Autoencoder
from sklearn.utils.extmath import cartesian
import matplotlib.pyplot as plt
from src.utils.task_dictionary import task_dict
import pickle

task_name = 'cartpole'
model_name = 'very-simple'  # which auto-encoder to use

env = gym.make(task_dict[task_name])
env.reset()  # returns initial state


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


def getState(stateBins, features):
    # print(len(stateBins))
    # print(len(features))
    assert len(stateBins) == len(features)
    bin = [np.where(stateBins[i] <= f)[0][-1] for i, f in enumerate(features)]
    # stateIndex = feature1*binSize^numRestFeatures + feature2*binSize^numRestFeatures...
    stateIndex = int(np.sum(bin * np.power(len(stateBins[0]), np.linspace(len(features)-1, 0, len(features)))))
    return stateIndex


# get shape of saved weights
net_weights = torch.load('../../models/' + model_name + '.model')
encoder_shape = list(net_weights.values())[0].size()
decoder_shape = list(net_weights.values())[len(net_weights)-1].size()

# init autoencoder according to size of loaded weights
net = Autoencoder(encoder_shape[1], encoder_shape[0], decoder_shape[0])
net.load_state_dict(torch.load('../../models/' + model_name + '.model'))


num_features = net.encoder.out_features

# in CartPole-v0, parameters are:
# [position of cart, velocity of cart, angle of pole, rotation rate of pole]
# print('low', env.observation_space.low)
# print('high', env.observation_space.high)
bins = 50
# stateBins = binThese(env.observation_space.low, env.observation_space.high, bins=bins, cutOffInf=5)
stateBins = binModel(0, 1, num_features, bins=bins)
allStates = cartesian(stateBins)  # round

# print(stateBins)
print(allStates[0, :])

qValues = np.zeros([len(allStates), env.action_space.n])
totalReward = 0
alpha = 0.618
GAMMA = 0.995
total_episode = 10000
print(f'qValues.shape: {qValues.shape}')

for episode in range(1, total_episode):
    done = False
    totalReward, reward = 0, 0
    state = env.reset()
    explore_rate = (1 - episode / total_episode) / 10 + 0.01
    # here we need to find the state of all states
    # transform to encoder latent representation
    # TODO: in case of sas.model, also need to append action
    state = net.activation(net.encoder(torch.tensor(state).float())).tolist()
    # print(f'latent: {state}')
    stateInd = getState(stateBins, state)
    updateQ = []
    while not done:
        if np.random.rand() > explore_rate:  # exploit
            action = np.argmax(qValues[stateInd])  # 1
        else:  # explore
            action = np.random.randint(0, len(qValues[0]))
        # print(f'action: {action}')
        internalRepr, reward, done, info = env.step(action)  # 2
        totalReward += reward
        latentRepr = net.activation(net.encoder(torch.tensor(internalRepr).float())).tolist()
        indState2 = getState(stateBins, latentRepr)  # find the state bin
        # qValues[stateInd, action] += alpha * (reward + np.max(qValues[indState2]) - qValues[stateInd, action])  # 3
        qValues[stateInd, action] = reward + GAMMA * np.max(qValues[indState2])  # 3
        # add state and action for later backtracking
        updateQ.append([stateInd, action])
        stateInd = indState2
        if done:
            updateQ.pop()
        # print(stateInd)
    # update the qValues that led to that state
    while len(updateQ) > 0:
        oldStateInd, oldaction = updateQ.pop()
        qValues[oldStateInd, oldaction] = reward + GAMMA * np.max(qValues[stateInd])
        stateInd, action = oldStateInd, oldaction
    if episode % 50 == 0:
        print('Episode {}, explore rate {}, Total Reward: {}'.format(episode, round(explore_rate, 2), totalReward))

print(np.sum(qValues)/np.count_nonzero(qValues))

print(np.max(qValues, axis=-1))

print('Saving q-values...')
with open('_'.join(['../../data/q_vals', task_name, model_name]) + '.pkl', 'wb') as f:
    pickle.dump(qValues, f)


print('Testing...')
done = False
state = env.reset()
test_total_reward = 0
while not done:
    env.render()
    state = net.activation(net.encoder(torch.tensor(state).float())).tolist()
    stateInd = getState(stateBins, state)
    # stateInd = np.where(allStates == state)[0][0]
    action = np.argmax(qValues[stateInd])
    state, reward, done, _ = env.step(action)
    test_total_reward += reward
print("Ended with reward: ", test_total_reward)

env.close()