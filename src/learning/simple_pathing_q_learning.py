from src.gym_custom_tasks.envs.simplePathing import SimplePathing
from src.utils.exploration import boltzmann_explore

# PARAMETERS
OBSTACLES = False
T = 3
gamma = 0.999
episodes = 10000

# ENVIRONMENT
env = SimplePathing(30, 30, False)

# TRAINING
q_table = [[[0 for _ in range(env.action_space.n)] for _ in range(env.width)] for _ in range(env.height)]
total_reward = 0
for episode in range(episodes):
    observation = env.reset()

    timesteps = 0
    done = False
    while not done:
        state = observation

        # choose action with exploration
        action = boltzmann_explore(q_table[state[1]][state[0]], T)

        # observe
        observation, reward, done, _ = env.step(action)  # observation is pos, velo
        total_reward += reward

        # update
        state_value = max(q_table[observation[1]][observation[0]])
        q_table[state[1]][state[0]][action] = reward + gamma * state_value

        timesteps += 1

        if episode % 1000 == 0 and episode >= 4000:
            # env.render()
            pass

    if episode % 100 == 0:
        print(f"Epochs done: {str(episode)}; Avg. Reward: {total_reward/100}")
        total_reward = 0

# TESTING
print("TESTING")
observation = env.reset()
done = False
timesteps = 0
total_reward = 0
while not done:
    action = max(list(enumerate((q_table[observation[1]][observation[0]]))), key=lambda x: x[1])[0]
    observation, reward, done, _ = env.step(action)  # observation is pos, velo
    timesteps += 1
    total_reward += reward
    env.render()

env.show_breadcrumbs = True
print(env.get_pixelbased_representation())

print(f"Finished in {timesteps} time steps using final policy getting {total_reward}.")
