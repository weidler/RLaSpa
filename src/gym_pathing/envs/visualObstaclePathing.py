from src.gym_pathing.envs.simplePathing import SimplePathing
from src.gym_pathing.envs.obstaclePathing import ObstaclePathing


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