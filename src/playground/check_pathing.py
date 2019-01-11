import gym
import src.gym_pathing
import matplotlib.pyplot as plt

def plot_env(env):
    img = env.get_pixelbased_representation()
    plt.imshow(img, cmap="binary", origin="upper")
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.show()

env = gym.make('VisualObstaclePathing-v0')
env.reset()

# env.get_pixelbased_representation()
# plot_env(env)
# env.render()
for _ in range(15):
    env.render()
    env.step(1)
for _ in range(15):
    env.render()
    env.step(0)

observation, reward, done, _ = env.step(1)

print('Observation:', type(observation), 'size:', observation.shape)
print('Reward:', type(reward), 'reward-value:', reward)

plot_env(env)
# env.render()

# plt.show()