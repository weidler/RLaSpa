import copy
import math
import pprint
import random

import numpy

from src.task.Task import Task

class SimplePathing(Task):
    BACKGROUND_SYMBOL = "."
    BACKGROUND_PIXEL = 0

    TARGET_SYMBOL = "X"
    TARGET_PIXEL = 0.8

    AGENT_SYMBOL = "A"
    AGENT_PIXEL = 0.3

    PADDING = 10

    DEFAULT_REWARD = -10
    TARGET_REWARD = 10

    def __init__(self, width: int, height: int):
        if not (width > SimplePathing.PADDING * 2 and height > SimplePathing.PADDING * 2):
            raise ValueError(
                "Width and Height need to be larger than double the padding of the environment, hence > {0}.".format(
                    SimplePathing.PADDING * 2))

        self.width = width
        self.height = height

        self.action_space = [0, 1, 2, 3]  # UP, RIGHT, DOWN, LEFT

        self.target_coords = [random.randint(SimplePathing.PADDING, self.width - SimplePathing.PADDING),
                              random.randint(SimplePathing.PADDING, self.height - SimplePathing.PADDING)]
        self.static_map = self._generate_static_map()

        self.start_state = [width // 2, height // 2]
        self.current_state = self.start_state.copy()

        self.state_trail = []

        self.show_breadcrumbs = True

    def __repr__(self):
        representation = ""
        if self.show_breadcrumbs:
            view = self._get_current_view_with_trail()
        else:
            view = self._get_current_view()

        for row in view:
            representation += " ".join(map(str, row)) + "\n"
        return representation

    def _generate_static_map(self):
        static_map = [[SimplePathing.BACKGROUND_SYMBOL for _ in range(self.width)] for _ in range(self.height)]
        static_map[self.target_coords[1]][self.target_coords[0]] = SimplePathing.TARGET_SYMBOL

        return static_map

    def _get_current_view(self):
        view = copy.deepcopy(self.static_map)
        if view[self.current_state[1]][self.current_state[0]] != "X":
            view[self.current_state[1]][self.current_state[0]] = "A"
        else:
            view[self.current_state[1]][self.current_state[0]] = "Ä"
        return view

    def _get_current_view_with_trail(self):
        view = self._get_current_view()
        for step in range(-1, -len(self.state_trail), -1):
            state = self.state_trail[step]
            view[state[1]][state[0]] = "O"
        return view

    def step(self, action: int):
        next_state = self.current_state.copy()
        self.state_trail.append(self.current_state)

        if action == 0:
            next_state[1] = max(0, next_state[1] - 1)
        elif action == 1:
            next_state[0] = min(next_state[0] + 1, self.width - 1)
        elif action == 2:
            next_state[1] = min(next_state[1] + 1, self.height - 1)
        elif action == 3:
            next_state[0] = max(0, next_state[0] - 1)

        reward = SimplePathing.DEFAULT_REWARD
        done = False
        if next_state == self.target_coords:
            reward = SimplePathing.TARGET_REWARD
            done = True

        self.current_state = next_state.copy()
        return next_state, reward, done

    def reset(self):
        self.current_state = self.start_state.copy()
        self.state_trail = []
        return self.current_state

    def get_pixelbased_representation(self):
        view = self._get_current_view()
        pixels = copy.deepcopy(view)
        for y, row in enumerate(view):
            for x, symbol in enumerate(row):
                pixel = SimplePathing.BACKGROUND_PIXEL
                if x == self.current_state[0] and y == self.current_state[1]: pixel = SimplePathing.AGENT_PIXEL
                if symbol == SimplePathing.TARGET_SYMBOL: pixel = SimplePathing.TARGET_PIXEL
                pixels[y][x] = pixel

        return pixels

if __name__ == "__main__":
    env = SimplePathing(30, 30)

    for dir in range(0, 4):
        for i in range(10):
            s, _, _ = env.step(dir)

    numpy.set_printoptions(linewidth=150)
    env.visualize()
