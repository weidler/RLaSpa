import pickle

import gym
import numpy as np

gym.logger.set_level(40)
env = gym.make('CartPole-v1')

bestLength = 0
episode_Length = []

best_weights = np.zeros(4)

history = []  # History with all the states, actions and rewards for the different epochs
for i in range(100):
    new_weights = np.random.uniform(-1.0, 1.0, 4)

    length = []
    for j in range(100):
        observation = env.reset()

        done = False
        count = 0
        while not done:
            # comment this one out for real training
            # env.render()
            count += 1
            action = 1 if np.dot(observation, new_weights) > 0 else 0
            observation, reward, done, _ = env.step(action)
            history.append([observation, action, reward])
        length.append(count)
    average_length = float(sum(length) / len(length))
    if average_length > bestLength:
        bestLength = average_length
        best_weights = new_weights
    episode_Length.append(average_length)
    if i % 10 == 0:
        print("Best length is ", bestLength)

# Save list with epochs
with open("../../data/cartpole.pkl", "wb") as f:
    pickle.dump(history, f)

# Learning finished --> play with the best weights
done = False
count = 0
observation = env.reset()

while not done:
    env.render()
    count += 1
    action = 1 if np.dot(observation, best_weights) > 0 else 0
    observation, reward, done, _ = env.step(action)

print("with best weights game lasted ", count, "moves")

env.close()

try:
    del env
except ImportError:
    pass
