import numpy as np
import os
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.data import rollout, types
from tqdm import tqdm, trange
from pathlib import Path
import datasets

from rule_based_solver import MinesweeperLogic # pyright: ignore[reportMissingImports]
from env_setup import create_flat_minesweeper_env  # pyright: ignore[reportMissingImports]
# --- Configuration ---
EPISODES_TO_GENERATE = 5000
OUTPUT_DATA_PATH = "initial_expert_trajectories.pkl"
ENV_ID = lambda: create_flat_minesweeper_env(16, 16, 40) 
N_ENVS = 16
EPISODE_PRINT_FREQ = 100
MINES_LEFT = 40
ENUM_LIMIT = 30

# In generate_rollouts.py, define this class near the top
class EpisodeTracker:
    """
    Custom stopping condition wrapper to track and display progress
    in the imitation.rollout.rollout function.
    """
    def __init__(self, min_episodes: int, print_frequency: int):
        self.min_episodes = min_episodes
        self.print_frequency = print_frequency
        self.episode_count = 0
        self.total_episodes = 0
        self.progress_bar = tqdm(total=min_episodes, desc="Generating Rollouts")

    def __call__(self, trajectories):
        """Called by rollout.rollout after each batch."""
        
        # Get the current number of completed episodes
        new_count = len(trajectories)
        
        if new_count > self.total_episodes:
            # Update the progress bar and print status
            self.progress_bar.update(new_count - self.total_episodes)
            self.total_episodes = new_count

            if new_count % self.print_frequency == 0:
                print(f"\n--- Progress: {new_count} of {self.min_episodes} episodes completed. ---")

        # Return the original stopping condition
        return new_count >= self.min_episodes

def save_transitions_legacy(path, transitions):
    data_dict = {
        "obs": transitions.obs,
        "acts": transitions.acts,
        "next_obs": transitions.next_obs,
        "dones": transitions.dones,
        "infos": transitions.infos,
    }

    dataset = datasets.Dataset.from_dict(data_dict)
    dataset.save_to_disk(str(Path(path)))

def get_expert_actions(obs: np.ndarray, state=None, dones=None) -> np.ndarray:
    """
    Takes a batch of observations (boards) and returns the corresponding batch 
    of optimal actions using the MinesweeperLogic solver.
    """
    batch_size = obs.shape[0]
    actions = np.zeros(batch_size, dtype=np.int64)
    board_H, board_W = obs.shape[1], obs.shape[2]
    
    for i in range(batch_size):
        
        # 1. CRITICAL FIX: Cast Observation back to Integer Dtype
        # This converts the float array from DummyVecEnv back to int32 for the solver.
        board_state = obs[i].astype(np.int32).reshape(board_H, board_W)
        
        # 2. Instantiate and Solve (Fixes C: Enum Limit)
        solver = MinesweeperLogic(board_state, mines_left=MINES_LEFT) 
        safe_moves, mine_moves, prob_dict = solver.solve(enum_limit=ENUM_LIMIT) 
        
        # 3. Action Selection Logic
        if safe_moves:
            # Choose a guaranteed safe move (highest priority)
            r, c = safe_moves[0] 
        elif prob_dict:
            # Choose the square with the lowest mine probability
            best_move = min(prob_dict.items(), key=lambda item: item[1])
            r, c = best_move[0]
        else:
            # Fallback to random guess among unopened cells
            unknown_cells = np.argwhere(board_state == -1)
            if len(unknown_cells) > 0:
                idx = np.random.randint(len(unknown_cells))
                r, c = unknown_cells[idx]
            else:
                r, c = (0, 0) # Failsafe
                
        # 4. Flatten Action and Store (The format needed by your FlatActionWrapper)
        actions[i] = r * board_W + c
                
    return actions, None

# --- 2. The Rollout Function ---
def generate_expert_data():
    """Performs rollouts using the expert logic and saves the trajectories."""
    
    # 1. Setup Vectorized Environment
    venv = DummyVecEnv([ENV_ID] * N_ENVS)
    SEED = 42
    rng = np.random.default_rng(SEED)

    print(f"Generating expert data: Rolling out {EPISODES_TO_GENERATE} episodes...")

    # Initialize the custom tracker object
    tracker = EpisodeTracker(
        min_episodes=EPISODES_TO_GENERATE,
        print_frequency=EPISODE_PRINT_FREQ
    )

    # 2. Perform Rollouts
    # rollout.rollout runs the provided policy (our expert function) in the environment.
    trajectories = rollout.rollout(
        policy=get_expert_actions,
        venv=venv,
        sample_until=tracker,
        rng=rng,
    )
    tracker.progress_bar.close()

    print(f"Rollout finished. Collected {len(trajectories)} trajectories.")

    # 3. Convert and Save Data
    # The 'imitation' library requires a Transitions object for DAgger.
    transitions = rollout.flatten_trajectories(trajectories)
    # Save the Transitions object to the specified path
    save_transitions_legacy(OUTPUT_DATA_PATH, transitions)

    print(f"Data saved successfully to {OUTPUT_DATA_PATH}")

if __name__ == "__main__":
    # You would execute this function to create the .pkl file
    generate_expert_data()
    print("Rollout function defined. Execute `generate_expert_data()` after ensuring dependencies are met.")