from env_setup import create_small_env

num_episodes = 1

for ep in range(num_episodes):
    env = create_small_env()
    obs, info = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        action = env.action_space.sample()  # random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
        print(f"obs {obs} \n, Action={action}, Reward={reward} \n")
    print(f"Episode {ep+1}, Total Reward = {total_reward}, Steps = {steps}")
    print(f"final_state: {env.unwrapped.state}")
    # print(f"bombs: {info['map']}")

