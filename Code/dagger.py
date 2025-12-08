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

height = 16
width = 16
board_size = height * width

def map_idx_to_coords(idx, width = width):
    row = idx // width
    col = idx % width

    return np.array([row, col])

def map_coords_to_idx(coords, width = width):
    if isinstance(coords, np.ndarray):
        coords = coords.tolist()

    return coords[0] * width + coords[1]

# MLP policy network
class MinesweeperPolicy(nn.Module):
    def __init__(self, input_size, output_size):
        super(MinesweeperPolicy, self).__init__()
        hidden_size = 512
        
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),  
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        return self.net(x)

# Load in data from file
def load_and_format_expert_data(arrow_folder = 'initial_expert_data'):
    print(f'Loading initial expert data from: {arrow_folder}')
    dataset = Dataset.load_from_disk(arrow_folder)

    all_obs = dataset['obs']
    all_acts = dataset['acts']
    print(f'Loaded initial expert dataset with {len(all_obs)} observations')

    return all_obs, all_acts 

def train_policy_on_dagger_data(policy_net, all_dagger_obs, all_dagger_acts, epochs = 5, batch_size = 64, lr = 1e-4):
    states_tensor = torch.tensor(all_dagger_obs, dtype=torch.float32)
    states_tensor = (states_tensor + 1.0) / 9.0  

    actions_tensor = torch.tensor(all_dagger_acts, dtype=torch.long)

    dagger_dataset = TensorDataset(states_tensor, actions_tensor)
    dagger_dataloader = DataLoader(dagger_dataset, batch_size = batch_size, shuffle = True)
    optimizer = optim.Adam(policy_net.parameters(), lr = lr) 
    criterion = nn.CrossEntropyLoss()
    
    policy_net.train() 
    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        for obs_batch, acts_batch in dagger_dataloader:
            optimizer.zero_grad()
            obs_batch_flat = obs_batch.view(obs_batch.size(0), -1) 
            
            logits = policy_net(obs_batch_flat)
            loss = criterion(logits, acts_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Calculate accuracy
            _, predicted_actions = torch.max(logits.data, 1)
            total_samples += acts_batch.size(0)
            correct_predictions += (predicted_actions == acts_batch).sum().item()

        avg_loss = total_loss / len(dagger_dataloader)
        accuracy = 100 * correct_predictions / total_samples
        print(f'Training Epoch {epoch + 1} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%')
        
    policy_net.eval()



def run_dagger(policy_net, env, rule_based_solver, initial_obs, initial_acts, dagger_iterations = 10, num_rollouts_per_iter = 100, initial_epochs = 5, retraining_epochs = 2):
    all_dagger_obs = list(initial_obs)
    all_dagger_acts = list(initial_acts)

    print('Starting DAgger Process...') 

    # Initial Behavioral Cloning
    print(f'Initializing policy with Behavioral Cloning...')
    train_policy_on_dagger_data(policy_net, all_dagger_obs, all_dagger_acts, epochs = initial_epochs)

    for i in range(dagger_iterations):
        print(f'\nDAgger Iteration {i + 1} / {dagger_iterations}')
        new_obs_states_list = [] 

        # Run Policy and Collect New States
        policy_net.eval()
        for _ in range(num_rollouts_per_iter):
            obs, info = env.reset()
            done = False
            truncated = False
            
            while not done and not truncated:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                obs_tensor_normalized = (obs_tensor + 1.0) / 9.0 
                obs_tensor_flat = obs_tensor_normalized.view(obs_tensor_normalized.size(0), -1) 

                # Get action from policy
                with torch.no_grad():
                    action_logits = policy_net(obs_tensor_flat)
                    action_idx = torch.argmax(action_logits).item() 
                
                new_obs_states_list.append(obs) 
                action_coords = map_idx_to_coords(action_idx, width)
                
                # Take step
                obs, reward, done, truncated, info = env.step(action_coords)
            
        # Query Expert for Optimal Actions
        new_expert_acts_flat = []
        for state_2d in new_obs_states_list:
            expert_action_coords = rule_based_solver.deterministic_solver(state_2d)
            expert_action_idx = map_coords_to_idx(expert_action_coords, width)
            new_expert_acts_flat.append(expert_action_idx)
            
        # Aggregate Data 
        print(f'Collected {len(new_obs_states_list)} new state-action pairs')
        
        all_dagger_obs.extend(new_obs_states_list)
        all_dagger_acts.extend(new_expert_acts_flat)

        # Re-Train Policy using retraining epochs
        print(f'Total aggregated dataset size: {len(all_dagger_obs)}')
        train_policy_on_dagger_data(policy_net, all_dagger_obs, all_dagger_acts, epochs = retraining_epochs)
        
    print('\nDAgger process complete!')
    return policy_net

if __name__ == "__main__":
    initial_bc_epochs = 6
    retraining_epochs = 3  
    dagger_iterations = 10
    n_rollouts = 100

    env = env_setup.MinesweeperEnv(height = height, width = width, num_mines = 40)
    policy_net = MinesweeperPolicy(input_size = board_size, output_size = board_size)
    initial_obs, initial_acts = load_and_format_expert_data('initial_expert_data_new')
    
    # Run the DAgger Algorithm
    final_policy = run_dagger(
        policy_net = policy_net, 
        env = env, 
        rule_based_solver = rule_based_solver, 
        initial_obs = initial_obs, 
        initial_acts = initial_acts,
        dagger_iterations = dagger_iterations, 
        num_rollouts_per_iter = n_rollouts,
        initial_epochs = initial_bc_epochs,
        retraining_epochs = retraining_epochs
    )

    # Save model weights
    model_path = 'minesweeper_dagger_policy_new.pth'
    torch.save(final_policy.state_dict(), model_path)
    print(f'\nFinal policy weights saved to "{model_path}"')