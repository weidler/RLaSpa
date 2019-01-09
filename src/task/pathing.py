import copy

import numpy as np

from src.task.task import _Task


class SimplePathing(_Task):
    BACKGROUND_SYMBOL = "."
    BACKGROUND_PIXEL = 0

    TARGET_SYMBOL = "X"
    TARGET_PIXEL = 0.8

    AGENT_SYMBOL = "A"
    AGENT_PIXEL = 0.3

    PADDING = 3

    DEFAULT_REWARD = -10
    TARGET_REWARD = 10

    def __init__(self, width: int, height: int):
        if not (width > SimplePathing.PADDING * 2 and height > SimplePathing.PADDING * 2):
            raise ValueError(
                "Width and Height need to be larger than double the padding of the environment, hence > {0}.".format(
                    SimplePathing.PADDING * 2))

        # ATTRIBUTES
        self.width = width
        self.height = height
        self.action_space = [0, 1, 2, 3]  # UP, RIGHT, DOWN, LEFT
        self.max_steps = width*height
        self.steps = 0

        # STATES
        self.target_coords = [width - SimplePathing.PADDING, SimplePathing.PADDING]
        self.start_state = [SimplePathing.PADDING, height - SimplePathing.PADDING]
        self.current_state = self.start_state.copy()

        # TRAIL
        self.state_trail = []
        self.show_breadcrumbs = True

        # MAP
        self.static_map = self._generate_static_map()
        self.static_pixels = self._generate_pixelbased_representation()

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
        # static_map = [[SimplePathing.BACKGROUND_SYMBOL for _ in range(self.width)] for _ in range(self.height)]
        # static_map[self.target_coords[1]][self.target_coords[0]] = SimplePathing.TARGET_SYMBOL
        static_map = np.full((self.height, self.width), SimplePathing.BACKGROUND_SYMBOL)
        static_map[self.target_coords[1], self.target_coords[0]] = SimplePathing.TARGET_SYMBOL

        # print(static_map)
        # print(static_map)
        return static_map

    def _get_current_view(self):
        view = copy.deepcopy(self.static_map)
        if view[self.current_state[1], self.current_state[0]] != "X":
            view[self.current_state[1], self.current_state[0]] = "A"
        else:
            view[self.current_state[1], self.current_state[0]] = "Ä"
        return view

    def _get_current_view_with_trail(self):
        view = self._get_current_view()
        for step in range(-1, -len(self.state_trail), -1):
            state = self.state_trail[step]
            view[state[1], state[0]] = "O"
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
        self.steps += 1
        done = self.steps >= self.max_steps
        if next_state == self.target_coords:
            reward = SimplePathing.TARGET_REWARD
            done = True

        self.current_state = next_state.copy()
        return next_state, reward, done, None  # returns None at pos 4 to match gym envs

    def reset(self):
        self.current_state = self.start_state.copy()
        self.state_trail = []
        self.steps = 0

        return self.current_state

    def _generate_pixelbased_representation(self):
        view = self._get_current_view()
        pixels = np.zeros(view.shape)
        pixels[view == SimplePathing.TARGET_SYMBOL] = SimplePathing.TARGET_PIXEL

        return pixels

    def get_pixelbased_representation(self):
        pixels = self.static_pixels.copy()
        pixels[self.current_state[1], self.current_state[0]] = SimplePathing.AGENT_PIXEL

        return pixels


class ObstaclePathing(SimplePathing):
    OBSTACLE_SYMBOL = "#"
    OBSTACLE_PIXEL = 1

    def __init__(self, width: int, height: int, obstacles: list):
        """
        :param width:
        :param height:
        :param obstacles:       list of lists where each sublist is [x_from, x_to, y_from, y_to]
        """
        self.obstacles = []
        super(ObstaclePathing, self).__init__(width, height)

        # create obstacles
        self.blocked_coordinates = []
        self.add_obstacles(obstacles)

        self.static_map = self._generate_static_map()
        self.static_pixels = self._generate_pixelbased_representation()

    def add_obstacles(self, obstacles):
        self.obstacles.extend(obstacles)
        for obst in obstacles:
            for y in range(obst[2], obst[3]):
                for x in range(obst[0], obst[1]):
                    self.static_map[y][x] = ObstaclePathing.OBSTACLE_SYMBOL
                    self.blocked_coordinates.append([x, y])

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

        # get blocked at coordinates
        if next_state in self.blocked_coordinates:
            next_state = self.current_state.copy()

        reward = SimplePathing.DEFAULT_REWARD
        self.steps += 1
        done = self.steps >= self.max_steps
        if next_state == self.target_coords:
            reward = SimplePathing.TARGET_REWARD
            done = True

        self.current_state = next_state.copy()
        return next_state, reward, done, None  # returns None at pos 4 to match gym envs

    def _generate_pixelbased_representation(self):
        pixels = super(ObstaclePathing, self)._generate_pixelbased_representation()
        for obst in self.obstacles:
            for y in range(obst[2], obst[3]):
                for x in range(obst[0], obst[1]):
                    pixels[y, x] = ObstaclePathing.OBSTACLE_PIXEL

        return pixels


class VisualObstaclePathing(ObstaclePathing):
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

        # get blocked at coordinates
        if next_state in self.blocked_coordinates:
            next_state = self.current_state.copy()

        reward = SimplePathing.DEFAULT_REWARD
        self.steps += 1
        done = self.steps >= self.max_steps
        if next_state == self.target_coords:
            reward = SimplePathing.TARGET_REWARD
            done = True

        self.current_state = next_state.copy()
        return self.get_pixelbased_representation(), reward, done, None

    def reset(self):
        self.current_state = self.start_state.copy()
        self.state_trail = []
        self.steps = 0
        return self.get_pixelbased_representation()


if __name__ == "__main__":
    env = ObstaclePathing(30, 30,
                          [[0, 18, 18, 21],
                           [21, 24, 10, 30]]
                          )
    env.visualize()
