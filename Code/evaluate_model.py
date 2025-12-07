# --- EVALUATION SCRIPT FIX (Option A) ---
import time
import numpy as np
import env_setup
from stable_baselines3 import PPO # <-- NEW: Need PPO for loading
import torch

import rule_based_solver
# IMPORTANT: You must also import or define the ChannelFirstWrapper and make_minesweeper_env 
# used during training to set up the environment for loading the model.

# 1. Load trained policy (Full SB3 model)
HEIGHT, WIDTH = 16, 16

# Load the full SB3 model object (usually a .zip file)
MODEL_PATH = "ppo_minesweeper_trained.zip" 

# You need a dummy environment to load the model (must match the training env)
# Assuming EVAL_ENV is set up correctly with the ChannelFirstWrapper
# from the training script.
# EVAL_ENV = make_vec_env(make_minesweeper_env, n_envs=1, wrapper_class=ChannelFirstWrapper) 

model = PPO.load(MODEL_PATH, device="cpu") # Load the complete PPO object

# --- 2. Helper: policy selects move (Corrected) ---
def policy_move(model, state):
    # Flatten the state (16, 16) -> (256) and add batch dimension (1, 256)
    obs = state.flatten()[None, ...].astype(np.float32)

    with torch.no_grad():
        # model.predict returns the action index array (e.g., shape (1,) or (1, 1))
        action_idx_arr, _ = model.predict(obs, deterministic=True)

        return action_idx_arr[0]
    

# 3. Evaluation loop (use 'model' instead of 'policy')
# ... rest of your evaluation logic ...
def find_nearest_legal_move(state, move, height, width):
    """
    Finds the unclicked cell closest to the predicted (but illegal) move.
    
    Args:
        state (np.ndarray): The current Minesweeper board state (H, W).
        illegal_move (tuple): The (row, col) predicted by the policy.
        height (int), width (int): Board dimensions.
        
    Returns:
        tuple: The (row, col) of the nearest legal (unclicked) cell, 
               or None if no legal moves exist.
    """
    pred_c, pred_r = move
    
    # 1. Identify all legal (unrevealed) coordinates
    # Assumes unrevealed cells have state < 0
    legal_c_coords, legal_r_coords = np.where(state < 0)
    
    if len(legal_c_coords) == 0:
        return None # No legal moves left
        
    min_dist_sq = float('inf')
    nearest_move = None
    
    # 2. Iterate through all legal moves and calculate squared distance
    # (Using squared distance avoids the expensive square root calculation)
    for c, r in zip(legal_c_coords, legal_r_coords):
        # Calculate squared Euclidean distance
        dist_sq = (r - pred_r)**2 + (c - pred_c)**2
        
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            nearest_move = (c, r)
            
    return nearest_move

def evaluate(env, model, n_games=10): # Pass model to evaluate
    # ... (evaluation logic remains the same, replacing policy with model) ...
    expert_success = 0
    policy_success = 0
    expert_times = []
    policy_times = []

    # --- Policy evaluation ---
    # NOTE: You MUST use the policy's move to advance the state, 
    #       not the expert's move (ex_move), or you are not testing the policy.
    for _ in range(n_games):
        state, _ = env.reset()
        done = False
        start_time = time.time()
        taken = set()
        taken2 = 0
        s2 = 0
        while not done:
            # 1. Get the move from the policy
            move = policy_move(model, state)
            # move = move[::-1]
            legal_move = find_nearest_legal_move(state, move, HEIGHT, WIDTH)
            # print(f'{move}, {legal_move}')
            # ex_move = rule_based_solver.deterministic_solver(state)
            # taken2 += 1
            taken.add(legal_move[1] * 16 + legal_move[0])
            
            # Optional: Get expert move for comparison/debugging
            # ex_move = rule_based_solver.deterministic_solver(state)
            # print(f"Policy Move: {move}, Expert Move: {ex_move}")
            
            # 2. Use the POLICY's move for the step!
            state, reward, done, truncated, info = env.step(legal_move) 
            
            if done or truncated:
                break
        # ... (timing and success tracking) ...
        policy_times.append(time.time() - start_time)
        policy_success += len(taken)
        s2 += taken2

    # env.render() # Use close instead of render if you aren't rendering a lot
    print(f"Avg Steps: {policy_success / n_games}, Avg Time: {np.mean(policy_times)}")

if __name__ == "__main__":
    env = env_setup.MinesweeperEnv(height=HEIGHT, width=WIDTH, num_mines=40)
    evaluate(env, model, n_games=1000) # Pass model here
    # input('')