import gym
import numpy as np
from sklearn.utils.extmath import cartesian

# env = gym.make("Taxi-v2")
env = gym.make("CartPole-v0")
env.reset()  # returns initial state

# We can determine the total number of possible states using the following command:
# env.observation_space.n
# If you would like to visualize the current state, type the following:
# env.render()
# let's explore the actions available to the agent.
env.action_space.n


def binThese(lowerBound, upperBound, bins=10, cutOffInf=10000):
    assert len(upperBound) == len(lowerBound)
    numFeatures = len(upperBound)
    featureBins = np.zeros((numFeatures, bins+1))
    for i in range(numFeatures):
        lb = -cutOffInf if lowerBound[i] <= -cutOffInf else lowerBound[i]
        ub = cutOffInf if upperBound[i] >= cutOffInf else upperBound[i]
        featureBins[i] = np.histogram_bin_edges([lb, ub], bins=bins)
    return featureBins[:, :-1]


def getState(allStates, stateBins, features):
    assert len(stateBins) == len(features)
    bin = [np.where(stateBins[i] <= f)[0][-1] for i, f in enumerate(features)]
    # print(f'bin {bin}')
    # print(f'binIndex: {int(np.sum(bin * np.power(len(stateBins[0]), np.linspace(len(features)-1, 0, len(features)))))}')
    # stateIndex = feature1*binSize^numRestFeatures + feature2*binSize^numRestFeatures...
    stateIndex = int(np.sum(bin * np.power(len(stateBins[0]), np.linspace(len(features)-1, 0, len(features)))))
    return stateIndex


print(env.observation_space.low)
print(env.observation_space.high)
bins = 10
stateBins = binThese(env.observation_space.low, env.observation_space.high, bins=bins, cutOffInf=2.4)
allStates = cartesian(stateBins)
numFeatures = len(env.observation_space.low)

# print(stateBins)
print(allStates[0, :])

qValues = np.zeros([len(allStates), env.action_space.n])
totalReward = 0
alpha = 0.618
print(f'qValues.shape: {qValues.shape}')

for episode in range(1, 10001):
    done = False
    totalReward, reward = 0, 0
    state = env.reset()
    # here we need to find the state of all states
    stateInd = getState(allStates, stateBins, state)
    # stateInd = np.where(allStates == state)[0][0]
    # print(f'state {state}')
    # print(f'stateInd {stateInd}')
    # break
    # TODO map features to state
    while not done:
            action = np.argmax(qValues[stateInd])  # 1
            # print(f'action: {action}')
            internalRepr, reward, done, info = env.step(action)  # 2
            # print(reward)
            indState2 = getState(allStates, stateBins, internalRepr)
            # indState2 = np.where(allStates == state2)[0][0]
            qValues[stateInd, action] += alpha * (reward + np.max(qValues[indState2]) - qValues[stateInd, action])  # 3
            totalReward += reward
            stateInd = indState2
            # print(stateInd)
    if episode % 50 == 0:
        print('Episode {} Total Reward: {}'.format(episode, totalReward))

print(np.sum(qValues)/np.count_nonzero(qValues))

done = False
state = env.reset()
while not done:
    env.render()
    stateInd = getState(allStates, stateBins, state)
    # stateInd = np.where(allStates == state)[0][0]
    action = np.argmax(qValues[stateInd])
    state, reward, done, _ = env.step(action)
print("Ended with reward: ", reward)

env.close()
# # Random aka Stupid algorithm:
# state = env.reset()
# counter = 0
# reward = None
# while reward != 20:
#     state, reward, done, info = env.step(env.action_space.sample())
#     counter += 1
#
# print(counter)
