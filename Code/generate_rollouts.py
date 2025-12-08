import numpy as np
import os
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.data import rollout, types
from tqdm import tqdm, trange
from pathlib import Path
import datasets
import rule_based_solver 
import env_setup

EPISODES_TO_GENERATE = 5000
OUTPUT_DATA_PATH = 'initial_expert_trajectories'
ENV_ID = lambda: env_setup.create_flat_minesweeper_env(16, 16, 40) 
N_ENVS = 16
EPISODE_PRINT_FREQ = 100
MINES_LEFT = 40
ENUM_LIMIT = 30

class EpisodeTracker:
    def __init__(self, min_episodes: int, print_frequency: int):
        self.min_episodes = min_episodes
        self.print_frequency = print_frequency
        self.episode_count = 0
        self.total_episodes = 0
        self.progress_bar = tqdm(total = min_episodes, desc = 'Generating Rollouts')

    def __call__(self, trajectories):
        new_count = len(trajectories)
        
        if new_count > self.total_episodes:
            # Update the progress bar and print status
            self.progress_bar.update(new_count - self.total_episodes)
            self.total_episodes = new_count

            if new_count % self.print_frequency == 0:
                print(f'\nProgress: {new_count} of {self.min_episodes} episodes completed.')

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

def get_expert_actions(obs: np.ndarray, state = None, dones = None) -> np.ndarray:
    batch_size = obs.shape[0]
    actions = np.zeros(batch_size, dtype = np.int64)
    board_H, board_W = obs.shape[1], obs.shape[2]
    
    for i in range(batch_size):
        board_state = obs[i].astype(np.int32).reshape(board_H, board_W)
        solver = rule_based_solver.MinesweeperLogic(board_state, mines_left = MINES_LEFT) 
        safe_moves, mine_moves, prob_dict = solver.solve(enum_limit = ENUM_LIMIT) 
        
        # Action selection
        if safe_moves:
            # Choose a guaranteed safe move
            r, c = safe_moves[0] 
        elif prob_dict:
            # Choose square with lowest mine probability
            best_move = min(prob_dict.items(), key = lambda item: item[1])
            r, c = best_move[0]
        else:
            # Random guess fallback
            unknown_cells = np.argwhere(board_state == -1)
            if len(unknown_cells) > 0:
                idx = np.random.randint(len(unknown_cells))
                r, c = unknown_cells[idx]
            else:
                r, c = (0, 0)
        
        # Flattening
        actions[i] = r * board_W + c
                
    return actions, None

# Rollout function
def generate_expert_data():
    venv = DummyVecEnv([ENV_ID] * N_ENVS)
    SEED = 42
    rng = np.random.default_rng(SEED)

    print(f'Generating expert data: Rolling out {EPISODES_TO_GENERATE} episodes...')

    tracker = EpisodeTracker(
        min_episodes = EPISODES_TO_GENERATE,
        print_frequency = EPISODE_PRINT_FREQ
    )

    trajectories = rollout.rollout(
        policy = get_expert_actions,
        venv = venv,
        sample_until = tracker,
        rng = rng,
    )
    tracker.progress_bar.close()

    print(f'Rollout finished. Collected {len(trajectories)} trajectories')

    transitions = rollout.flatten_trajectories(trajectories)
    save_transitions_legacy(OUTPUT_DATA_PATH, transitions)
    print(f'Data saved successfully to {OUTPUT_DATA_PATH}')

if __name__ == "__main__":
    generate_expert_data()
    print('Rollout function defined. Execute `generate_expert_data()` after ensuring dependencies are met.')