import time
import numpy as np
import env_setup
from stable_baselines3 import PPO
import torch
import rule_based_solver

# Params
height, width = 16, 16
num_mines = 40
# baseline_path = 'ppo_intermediate_500k.zip'
model_path = 'ppo_minesweeper_trained_new.zip'
model = PPO.load(model_path, device = 'cpu')

# Select a move from the model based on board state
def policy_move(model, state):
    obs = state.flatten()[None, ...].astype(np.float32)

    with torch.no_grad():
        action_idx_arr, _ = model.predict(obs, deterministic=True)
        return action_idx_arr[0]
    

# Ensure model doesn't predict an already clicked cell
def find_nearest_legal_move(state, move, height, width):
    pred_c, pred_r = move

    # Get action space
    legal_c_coords, legal_r_coords = np.where(state < 0)
    
    if len(legal_c_coords) == 0:
        return None
        
    min_distance = float('inf')
    nearest_move = None
    
    # Distance formula
    for c, r in zip(legal_c_coords, legal_r_coords):
        distance = (r - pred_r)**2 + (c - pred_c)**2
        
        if distance < min_distance:
            min_distance = distance
            nearest_move = (c, r)
            
    return nearest_move

def evaluate_model(env, model, n_games=100):
    policy_times = []
    policy_moves_taken = 0
    average_rewards = []

    for _ in range(n_games):
        state, _ = env.reset()
        done = False
        current_reward = 0
        while not done:
            policy_start_time = time.time()
            move = policy_move(model, state)
            legal_move = find_nearest_legal_move(state, move, height, width)
            policy_moves_taken += 1
            # print(f'{move}, {legal_move}')
            policy_times.append(time.time() - policy_start_time)

            state, reward, done, truncated, info = env.step(legal_move) 
            current_reward += reward
            
            if done or truncated:
                break
        
        average_rewards.append(current_reward)

    print(f'Avg Policy Steps: {policy_moves_taken / n_games}, Avg Time Per Move: {np.mean(policy_times)}, Avg Reward: {np.mean(average_rewards)}')

def evaluate_expert(env, n_games=100):
    expert_times = []
    expert_moves_taken = 0
    average_rewards = []

    for _ in range(n_games):
        state, _ = env.reset()
        done = False
        current_reward = 0
        while not done:
            expert_start_time = time.time()
            expert_move = rule_based_solver.deterministic_solver(state)
            expert_moves_taken += 1
            expert_times.append(time.time() - expert_start_time)

            state, reward, done, truncated, info = env.step(expert_move) 
            current_reward += reward
            
            if done or truncated:
                break
        
        average_rewards.append(current_reward)

    print(f'Avg Expert Steps: {expert_moves_taken / n_games}, Avg Time Per Move: {np.mean(expert_times)}, Avg Reward: {np.mean(average_rewards)}')

if __name__ == "__main__":
    env = env_setup.MinesweeperEnv(height = height, width = width, num_mines = num_mines)
    evaluate_model(env, model, n_games = 10000)
    # evaluate_expert(env, n_games = 1000)
    input('')