import copy
import math
import pprint
import random


class SimplePathing(object):

    BACKGROUND_COLOR = "."
    TARGET_COLOR = "X"

    PADDING = 10

    DEFAULT_REWARD = -10
    TARGET_REWARD = 10

    def __init__(self, width: int, height:int ):
        if not (width > SimplePathing.PADDING * 2 and height > SimplePathing.PADDING * 2):
            raise ValueError("Width and Height need to be larger than double the padding of the environment, hence > {0}.".format(SimplePathing.PADDING * 2))

        self.width = width
        self.height = height

        self.action_space = [0, 1, 2, 3]  # UP, RIGHT, DOWN, LEFT

        self.target_coords = [random.randint(SimplePathing.PADDING, self.width - SimplePathing.PADDING), random.randint(SimplePathing.PADDING, self.height - SimplePathing.PADDING)]
        self.static_map = self._generate_static_map()

        self.start_state = [width // 2, height // 2]
        self.current_state = self.start_state.copy()

    def __repr__(self):
        representation = ""
        for row in self._get_current_view():
            representation += " ".join(map(str, row)) + "\n"
        return representation

    def _generate_static_map(self):
        static_map = [[SimplePathing.BACKGROUND_COLOR for _ in range(self.width)] for _ in range(self.height)]
        static_map[self.target_coords[0]][self.target_coords[1]] = SimplePathing.TARGET_COLOR

        return static_map

    def _get_current_view(self):
        view = copy.deepcopy(self.static_map)
        view[self.current_state[1]][self.current_state[0]] = "A"
        return view

    def step(self, action: int):
        next_state = self.current_state.copy()

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

if __name__ == "__main__":
    env = SimplePathing(30, 30)

    s = [2,3]
    for i in range(10):
        s, _, _ = env.step(random.randint(0, 3))
        print(env)