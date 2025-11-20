from env_setup import create_small_env
from gymnasium.utils.step_api_compatibility import convert_to_terminated_truncated_step_api

num_episodes = 10

for ep in range(num_episodes):
    env = create_small_env()
    obs = env.reset()
    info = {}
    done = False
    total_reward = 0
    steps = 0

    while not done:
        action = env.action_space.sample()  # random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
    print(f"Episode {ep+1}: Total Reward = {total_reward}, Steps = {steps}")
