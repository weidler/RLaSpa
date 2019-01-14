import gym
import torch
import numpy
import matplotlib.pyplot as plt

import src.gym_custom_tasks

class Evasion_walls(gym.Env):

    AGENT_PIXEL = 0.3
    OBSTACLE_PIXEL = 0.6

    def __init__(self, width: int=30, height: int=30, obstacle_chance: float=0.02):
        self.width = width
        self.height = height
        self.action_space = gym.spaces.Discrete(3) # UP, DOWN, STRAIGHT
        self.observation_space = gym.spaces.Box(low=0, high=1, dtype=numpy.float16, shape=(self.width, self.height))

        self.dungeon, self.pos_agent = self._init_dungeon()
        self.obstacle_chance = torch.empty(self.height, 1).new_full((self.height, 1), obstacle_chance)  # 0.1 = 10% chance that an obstacle is spawned
        self.max_steps = 500
        self.steps = 0

    def _init_dungeon(self):
        track = torch.zeros((self.height, self.width))
        pos = self.height//2
        track[pos, 0] = Evasion_walls.AGENT_PIXEL
        return track, pos

    def reset(self):
        self.steps = 0
        self.dungeon, self.pos_agent = self._init_dungeon()
        return self.dungeon

    def step(self, action):
        self.steps += 1
        done = self.steps >= self.max_steps
        # Straight does not change position of agent
        if action == 0:  # UP
            self.pos_agent = max(0, self.pos_agent-1)
        elif action == 1:  # DOWN
            self.pos_agent = min(self.height-1, self.pos_agent+1)

        # if agent crashed into obstacle --> over
        if self.dungeon[self.pos_agent, 1] != 0:
            done = True
            reward = 0
        else:
            reward = 10

        new_row = torch.zeros(self.height, 1)
        obstacle_pos = torch.bernoulli(self.obstacle_chance).byte()
        # obstacle_pos = torch.zeros(self.height, 1).byte()
        # obstacle_pos[15] = 1
        # obstacle_pos[8] = 1
        obst = obstacle_pos.nonzero()
        if len(obst) > 0:
            obst_min1 = obst[:, 0]-1
            obst_min1[obst_min1 < 0] = 0
            obst_plus1 = obst[:, 0]+1
            obst_plus1[obst_plus1 > self.height-1] = self.height-1
            obstacle_pos[obst_min1] = 1
            obstacle_pos[obst_plus1] = 1
        new_row[obstacle_pos] = Evasion_walls.OBSTACLE_PIXEL

        # add new row
        self.dungeon = torch.cat((self.dungeon[:, 1:], new_row), 1)
        self.dungeon[self.pos_agent, 0] = Evasion_walls.AGENT_PIXEL

        return self.dungeon, reward, done, None

    def render(self, mode='human', close=False):
        img = self.dungeon
        plt.clf()
        plt.imshow(img, cmap="binary", origin="upper")
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.draw()
        plt.pause(0.001)


if __name__ == "__main__":
    env = gym.make('EvasionWalls-v0')
    for _ in range(5):
        env.reset()
        done = False
        while not done:
            env.render()
            observation, reward, done, _ = env.step(2)
    # env.step(1)
    # observation, reward, done, _ = env.step(2)
    # print('Observation:', type(observation), 'size:', observation.shape)
    # print('Reward:', type(reward), 'reward-value:', reward)

    plt.imshow(env.dungeon, cmap="binary", origin="upper")
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.show()
