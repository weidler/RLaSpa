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
        self.obstacles = obstacles
        self.add_obstacles(obstacles)

    def add_obstacles(self, obstacles):
        self.obstacles.extend(obstacles)
        for obst in obstacles:
            for y in range(obst[2], obst[3]):
                for x in range(obst[0], obst[1]):
                    self.static_map[y][x] = ObstaclePathing.OBSTACLE_SYMBOL


    def get_pixelbased_representation(self):
        pixels = super(ObstaclePathing, self).get_pixelbased_representation()
        for obst in self.obstacles:
            for y in range(obst[2], obst[3]):
                for x in range(obst[0], obst[1]):
                    pixels[y][x] = ObstaclePathing.OBSTACLE_PIXEL

        return pixels

if __name__ == "__main__":
    env = ObstaclePathing(30, 30, [[4, 9, 3, 8], [14, 19, 20, 26]])

    env.visualize()