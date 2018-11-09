import gym
import numpy as np
from sklearn.utils.extmath import cartesian

# env = gym.make("Taxi-v2")
env = gym.make("CartPole-v1")
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


stateBins = binThese(env.observation_space.low, env.observation_space.high, bins=10, cutOffInf=5)
allStates = cartesian(stateBins)

print(stateBins)
print(allStates[0, :])

qValues = np.zeros([len(allStates), env.action_space.n])
totalReward = 0
alpha = 0.618

for episode in range(1, 1001):
    done = False
    totalReward, reward = 0, 0
    state = env.reset()
    # here we need to find the state of all states
    print(state)
    # TODO map features to state
    while not done:
            action = np.argmax(qValues[state])  # 1
            state2, reward, done, info = env.step(action)  # 2
            qValues[state, action] += alpha * (reward + np.max(qValues[state2]) - qValues[state, action])  # 3
            totalReward += reward
            state = state2
    if episode % 50 == 0:
        print('Episode {} Total Reward: {}'.format(episode, totalReward))


done = False
state = env.reset()
while not done:
    env.render()
    action = np.argmax(qValues[state])
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
