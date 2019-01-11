import gym
import torch
import numpy
import matplotlib.pyplot as plt

class Race(gym.Env):

    AGENT_PIXEL = 0.3
    DRIVER_PIXEL = 0.6

    def __init__(self):
        self.width = 20
        self.height = 30
        self.action_space = gym.spaces.Discrete(3) # LEFT, RIGHT, STRAIGHT
        self.observation_space = gym.spaces.Box(low=0, high=1, dtype=numpy.float16, shape=(self.width, self.height))

        self.track, self.pos_agent = self._init_track()
        self.driver_chance = torch.empty(1, self.width).new_full((1, self.width), 0.1)  # 0.1 = 10% chance that a driver is spawned
        self.max_steps = 500
        self.steps = 0

    def _init_track(self):
        track = torch.zeros((self.height, self.width))
        pos = self.width//2
        track[self.height-1, pos] = Race.AGENT_PIXEL
        return track, pos

    def reset(self):
        self.steps = 0
        self.track, self.pos_agent = self._init_track()
        return self.track

    def step(self, action):
        self.steps += 1
        done = self.steps >= self.max_steps
        # Straight does not change position of agent
        if action == 0:  # Left
            self.pos_agent = max(0, self.pos_agent-1)
        elif action == 1:  # Right
            self.pos_agent = min(self.width-1, self.pos_agent+1)

        # if agent crashed into obstacle --> over
        if self.track[-2, self.pos_agent] != 0:
            done = True
            reward = 0
        else:
            reward = 10

        new_row = torch.zeros(1, self.width)
        new_row[torch.bernoulli(self.driver_chance).byte()] = Race.DRIVER_PIXEL
        self.track = torch.cat((new_row, self.track[:-1]))
        self.track[-1, self.pos_agent] = Race.AGENT_PIXEL

        return self.track, reward, done, None

    def render(self, mode='human', close=False):
        img = self.track
        plt.clf()
        plt.imshow(img, cmap="binary", origin="upper")
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.draw()
        plt.pause(0.001)


if __name__ == "__main__":
    env = Race()

    done = False
    while not done:
        env.render()
        observation, reward, done, _ = env.step(2)
    # env.step(1)
    # observation, reward, done, _ = env.step(2)
    # print('Observation:', type(observation), 'size:', observation.shape)
    # print('Reward:', type(reward), 'reward-value:', reward)

    plt.imshow(env.track, cmap="binary", origin="upper")
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.show()