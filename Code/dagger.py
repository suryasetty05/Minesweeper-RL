import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datasets import Dataset
from torch.utils.data import TensorDataset, DataLoader
import os
import sys

import env_setup
import rule_based_solver 

# --- 2. Constants and Action Mapping Helpers ---

HEIGHT = 16
WIDTH = 16
BOARD_SIZE = HEIGHT * WIDTH # 256

def map_idx_to_coords(idx, width=WIDTH):
    row = idx // width
    col = idx % width
    return np.array([row, col])

def map_coords_to_idx(coords, width=WIDTH):
    if isinstance(coords, np.ndarray):
        coords = coords.tolist()
    return coords[0] * width + coords[1]


# --- 3. MLP Policy Network ---

class MinesweeperPolicy(nn.Module):
    def __init__(self, input_size, output_size):
        super(MinesweeperPolicy, self).__init__()
        # We keep the primary hidden size at 512
        hidden_size = 512
        
        # The network now has 4 Linear layers instead of 3
        self.net = nn.Sequential(
            # Input (256) -> Hidden 1 (512)
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            
            # Hidden 1 (512) -> Hidden 2 (512)
            nn.Linear(hidden_size, hidden_size),  
            nn.ReLU(),                            
            
            # Hidden 2 (512) -> Hidden 3 (256)
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            
            # Hidden 3 (256) -> Output (256)
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        return self.net(x)


# --- 4. Data Loading and Training Utilities ---
# The load_and_format_expert_data function is unchanged.

def load_and_format_expert_data(arrow_folder="initial_expert_data"):
    # ... (function body remains the same)
    print(f"Loading initial expert data from: {arrow_folder}")
    try:
        if not os.path.exists(arrow_folder):
            raise FileNotFoundError
            
        dataset = Dataset.load_from_disk(arrow_folder)
    except FileNotFoundError:
        print(f"Error: Expert data not found at '{arrow_folder}'.")
        print("Please ensure the folder is present and contains the 'datasets' files.")
        sys.exit(1)

    all_obs = dataset["obs"]
    all_acts = dataset["acts"]
    
    print(f"Loaded initial expert dataset with approx. {len(all_obs)} observations.")
    return all_obs, all_acts 

# --- REVISED FUNCTION: train_policy_on_dagger_data ---
def train_policy_on_dagger_data(policy_net, all_dagger_obs, all_dagger_acts, epochs=5, batch_size=64, lr=1e-4): # Reduced LR to 1e-4
    """
    Performs Behavioral Cloning (BC) on the aggregated dataset (D).
    
    FIX: Normalization is RE-INTRODUCED here to solve the input compression bottleneck.
    """
    if not all_dagger_obs or len(all_dagger_obs) == 0:
        print("No data to train on. Skipping training step.")
        return

    # 1. Convert to tensors (using raw [-1, 8] values)
    states_tensor = torch.tensor(all_dagger_obs, dtype=torch.float32)
    actions_tensor = torch.tensor(all_dagger_acts, dtype=torch.long)
    
    # 2. CRITICAL FIX: RE-INTRODUCING NORMALIZATION
    # Scales input range [-1, 8] to [0, 1]
    states_tensor = (states_tensor + 1.0) / 9.0  

    # --------------------------------------------------

    dagger_dataset = TensorDataset(states_tensor, actions_tensor)
    dagger_dataloader = DataLoader(dagger_dataset, batch_size=batch_size, shuffle=True)
    
    # Use the potentially reduced LR (defaulted to 1e-4 in the signature)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr) 
    criterion = nn.CrossEntropyLoss()
    
    policy_net.train() 
    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        for obs_batch, acts_batch in dagger_dataloader:
            optimizer.zero_grad()
            
            # Flatten the normalized observation batch: N x 16 x 16 -> N x 256
            obs_batch_flat = obs_batch.view(obs_batch.size(0), -1) 
            
            logits = policy_net(obs_batch_flat)
            loss = criterion(logits, acts_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # --- Accuracy Calculation ---
            _, predicted_actions = torch.max(logits.data, 1)
            total_samples += acts_batch.size(0)
            correct_predictions += (predicted_actions == acts_batch).sum().item()

        avg_loss = total_loss / len(dagger_dataloader)
        accuracy = 100 * correct_predictions / total_samples
        print(f"  Training Epoch {epoch+1} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
        
    policy_net.eval()


# --- 5. The DAgger Loop Execution ---

# --- REVISED FUNCTION: run_dagger ---
def run_dagger(policy_net, env, rule_based_solver, initial_obs, initial_acts, dagger_iterations=10, num_rollouts_per_iter=100, initial_epochs=5, retraining_epochs=2):
    
    try:
        all_dagger_obs = list(initial_obs)
        all_dagger_acts = list(initial_acts)
    except Exception as e:
        print(f"Error converting initial data columns to lists: {e}. Exiting.")
        sys.exit(1)

    print("\n--- Starting DAgger Process ---") 

    # 1. Initial Behavioral Cloning (BC) training: Using max epochs
    print(f"1. Initializing policy with Behavioral Cloning ({initial_epochs} epochs)...")
    # Note: lr is now defaulted to 1e-4 in the training function
    train_policy_on_dagger_data(policy_net, all_dagger_obs, all_dagger_acts, epochs=initial_epochs)

    for i in range(dagger_iterations):
        print(f"\n--- DAgger Iteration {i+1}/{dagger_iterations} ---")
        new_obs_states_list = [] 

        # 2. Run Policy (Rollouts) and Collect New States $s_t$
        policy_net.eval()
        for _ in range(num_rollouts_per_iter):
            obs, info = env.reset()
            done = False
            truncated = False
            
            while not done and not truncated:
                # Convert to tensor (RAW values)
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                
                # CRITICAL FIX: APPLY NORMALIZATION FOR POLICY INFERENCE
                # Scales input range [-1, 8] to [0, 1]
                obs_tensor_normalized = (obs_tensor + 1.0) / 9.0 

                # Flatten the normalized tensor from 1x16x16 to 1x256
                obs_tensor_flat = obs_tensor_normalized.view(obs_tensor_normalized.size(0), -1) 

                # Get action from the policy
                with torch.no_grad():
                    action_logits = policy_net(obs_tensor_flat)
                    action_idx = torch.argmax(action_logits).item() 
                
                # Store the raw 2D NumPy state before the step
                new_obs_states_list.append(obs) 
                
                # Convert flattened index back to [row, col] for env.step
                action_coords = map_idx_to_coords(action_idx, WIDTH)
                
                # Take a step
                obs, reward, done, truncated, info = env.step(action_coords)
            
        # 3. Query Expert for Optimal Actions ($a_{\text{expert}}$)
        new_expert_acts_flat = []
        for state_2d in new_obs_states_list:
            expert_action_coords = rule_based_solver.deterministic_solver(state_2d)
            expert_action_idx = map_coords_to_idx(expert_action_coords, WIDTH)
            new_expert_acts_flat.append(expert_action_idx)
            
        # 4. Aggregate Data 
        print(f"  Collected {len(new_obs_states_list)} new state-action pairs.")
        
        all_dagger_obs.extend(new_obs_states_list)
        all_dagger_acts.extend(new_expert_acts_flat)

        # 5. Re-Train Policy: Using increased retraining epochs
        print(f"  Total aggregated dataset size: {len(all_dagger_obs)}.")
        train_policy_on_dagger_data(policy_net, all_dagger_obs, all_dagger_acts, epochs=retraining_epochs)
        
    print("\nDAgger process complete! Policy trained.")
    return policy_net


# --- 6. Main Execution Block ---

if __name__ == "__main__":
    
    # --- AGGRESSIVE TRAINING PARAMETERS ---
    # Increased epochs slightly for better learning after normalization fix
    INITIAL_BC_EPOCHS = 6
    RETRAINING_EPOCHS = 3  
    DAGGER_ITERATIONS = 5
    NUM_ROLLOUTS = 100
    # --------------------------------------

    # Assume env_setup and rule_based_solver modules are in the path
    env = env_setup.MinesweeperEnv(height=HEIGHT, width=WIDTH, num_mines=40)
    policy_net = MinesweeperPolicy(input_size=BOARD_SIZE, output_size=BOARD_SIZE)
    
    initial_obs, initial_acts = load_and_format_expert_data()
    
    # Run the DAgger Algorithm with aggressive settings
    final_policy = run_dagger(
        policy_net=policy_net, 
        env=env, 
        rule_based_solver=rule_based_solver, 
        initial_obs=initial_obs, 
        initial_acts=initial_acts,
        dagger_iterations=DAGGER_ITERATIONS, 
        num_rollouts_per_iter=NUM_ROLLOUTS,
        initial_epochs=INITIAL_BC_EPOCHS,
        retraining_epochs=RETRAINING_EPOCHS
    )

    # Save the final model weights
    model_path = "minesweeper_dagger_policy_fixed.pth"
    torch.save(final_policy.state_dict(), model_path)
    print(f"\nFinal policy weights saved to '{model_path}'")