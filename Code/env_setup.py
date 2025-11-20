import gymnasium as gym
import gym_minesweeper
from gymnasium.utils.step_api_compatibility import convert_to_terminated_truncated_step_api

def create_minesweeper_env(height, width, num_mines):
    env = (gym.make("Minesweeper-v0",
                    height=height, 
                    width=width, 
                    num_mines=num_mines,
                    ))
    return env

def create_small_env():
    return create_minesweeper_env(5, 5, 5)
def create_beginner_env():
    return create_minesweeper_env(9, 9, 10)
def create_intermediate_env():  
    return create_minesweeper_env(16, 16, 40)