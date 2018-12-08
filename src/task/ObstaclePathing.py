from src.task.SimplePathing import SimplePathing

class ObstaclePathing(SimplePathing):

    OBSTACLE_SYMBOL = "#"
    OBSTACLE_PIXEL = 1

    def __init__(self, width: int, height: int, obstacles: list):
        """

        :param width:
        :param height:
        :param obstacles:       list of lists where each sublist is [x_from, x_to, y_from, y_to]
        """
        super(ObstaclePathing, self).__init__(width, height)

        # create obstacles
        self.obstacles = []
        self.blocked_coordinates = []
        self.add_obstacles(obstacles)

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
        done = False
        if next_state == self.target_coords:
            reward = SimplePathing.TARGET_REWARD
            done = True

        self.current_state = next_state.copy()
        return next_state, reward, done

    def get_pixelbased_representation(self):
        pixels = super(ObstaclePathing, self).get_pixelbased_representation()
        for obst in self.obstacles:
            for y in range(obst[2], obst[3]):
                for x in range(obst[0], obst[1]):
                    pixels[y][x] = ObstaclePathing.OBSTACLE_PIXEL

        return pixels

if __name__ == "__main__":
    env = ObstaclePathing(30, 30,
        [[0, 18, 18, 21],
         [21, 24, 10, 30]]
    )
    env.visualize()