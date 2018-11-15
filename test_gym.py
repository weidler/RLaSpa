from gym import wrappers
import gym
import numpy as np

gym.logger.set_level(40)
env = gym.make('CartPole-v1')


bestLength = 0
episode_Length = []

best_weights = np.zeros(4)

f = open("data/cartpole.data", "w")
for i in range(100):
    new_weights = np.random.uniform(-1.0, 1.0, 4)
    epoch_history = ""

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
            epoch_history += "{0}\t{1}\t{2}\n".format(list(observation), action, reward)
        # epoch_history += "YOU ARE A FAILURE\n"

        length.append(count)
    average_length = float(sum(length) / len(length))

    f.write(epoch_history)

    if average_length > bestLength:
        bestLength = average_length
        best_weights = new_weights
    episode_Length.append(average_length)
    if i % 10 == 0:
        print("Best length is ", bestLength)

f.close()

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
