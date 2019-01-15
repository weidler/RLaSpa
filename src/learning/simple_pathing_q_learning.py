from src.gym_custom_tasks.envs.simplePathing import SimplePathing
from src.utils.exploration import boltzmann_explore

# PARAMETERS
OBSTACLES = False
T = 3
gamma = 0.999
epochs = 10000

# ENVIRONMENT
env = SimplePathing(100, 100, False)


# TRAINING
q_table = [[[0 for _ in range(env.action_space.n)] for _ in range(env.width)] for _ in range(env.height)]
for epoch in range(epochs):
    observation = env.reset()

    timesteps = 0
    done = False
    while not done and timesteps <= 10000:
        state = observation

        # choose action with exploration
        action = boltzmann_explore(q_table[state[1]][state[0]], T)

        # observe
        observation, reward, done, _ = env.step(action)  # observation is pos, velo

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
    observation, reward, done, _ = env.step(action)  # observation is pos, velo
    timesteps += 1

env.show_breadcrumbs = True
print(env.get_pixelbased_representation())

print("Finished in {0} time steps using final policy.".format(timesteps))
