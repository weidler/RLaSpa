from src.task.pathing import SimplePathing, ObstaclePathing
from src.utils.exploration import boltzmann_explore

# PARAMETERS
OBSTACLES = False
T = 3
gamma = 0.999
epochs = 10000

# ENVIRONMENT
if not OBSTACLES:
    env = SimplePathing(100, 100)
else:
    env = ObstaclePathing(30, 30,
                          [[0, 13, 18, 20],
                           [16, 18, 11, 30],
                           [0, 25, 6, 8]]
                          )

    env.visualize()

# TRAINING
q_table = [[[0 for _ in range(len(env.action_space))] for _ in range(env.width)] for _ in range(env.height)]
for epoch in range(epochs):
    observation = env.reset()

    timesteps = 0
    done = False
    while not done and timesteps <= 10000:
        state = observation.copy()

        # choose action with exploration
        action = boltzmann_explore(q_table[state[1]][state[0]], T)

        # observe
        observation, reward, done = env.step(action)  # observation is pos, velo

        # update
        state_value = max(q_table[observation[1]][observation[0]])
        q_table[state[1]][state[0]][action] = reward + gamma * state_value

        timesteps += 1

    if epoch % 100 == 0: print("Epochs done: " + str(epoch))

# TESTING
observation = env.reset()
done = False
timesteps = 0
while not done:
    action = max(list(enumerate((q_table[observation[1]][observation[0]]))), key=lambda x: x[1])[0]
    observation, reward, done = env.step(action)  # observation is pos, velo
    timesteps += 1

env.show_breadcrumbs = True
print(env)

print("Finished in {0} time steps using final policy.".format(timesteps))
