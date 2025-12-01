import gymnasium as gym
import gym_minesweeper
from gymnasium.utils.step_api_compatibility import convert_to_terminated_truncated_step_api
import math
import numpy as np
import random

def create_minesweeper_env(height, width, num_mines):
    env = (gym.make("Minesweeper-v0",
                    height=height, 
                    width=width, 
                    num_mines=num_mines,
                    ))
    return env

if __name__ == '__main__':
    env = create_minesweeper_env(8, 8, 10)
    obs = env.reset()
    info = {}
    done = False
    total_reward = 0
    steps = 0

    action = [1, 1]
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    steps += 1

    if terminated:
        print('BOOM')
    print(f"Total Reward = {total_reward}, Steps = {steps}")
    env.render()
    input("enter to exit")
