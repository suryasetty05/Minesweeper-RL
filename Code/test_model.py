import rule_based_solver # pyright: ignore[reportMissingImports]
import numpy as np
from env_setup import create_minesweeper_env # pyright: ignore[reportMissingImports]
import time

def test_rule_based(episodes):
    sum_reward = 0
    sum_steps = 0
    wins = 0
    for ep in range(episodes):
        env = create_minesweeper_env(16, 16, 40)
        obs, info = env.reset()
        board = obs.copy()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action = rule_based_solver.deterministic_solver(board)
            obs, reward, terminated, truncated, info = env.step(action)
            board = obs.copy()
            total_reward += reward
            steps += 1
            if terminated:
                done = True
                break
        
        sum_reward += total_reward
        sum_steps += steps
        if total_reward >= 265: wins += 1

        
    return sum_reward/episodes, sum_steps/episodes, wins/episodes

if __name__ == '__main__':
    start_time = time.time()
    avg_reward, avg_steps, win_rate = test_rule_based(10)
    end_time = time.time()
    print(f"Average Reward: {avg_reward}, Average Steps: {avg_steps}, Win Rate: {win_rate}")
    print(f"Elapsed Time: {end_time - start_time} seconds")
    print(f"Time per Episode: {(end_time - start_time) / 10} seconds")
