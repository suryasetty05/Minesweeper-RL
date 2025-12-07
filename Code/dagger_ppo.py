import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import env_setup  # Your Minesweeper environment

# --- Environment Configuration ---
HEIGHT, WIDTH = 16, 16
NUM_MINES = 40
BOARD_SIZE = HEIGHT * WIDTH  # 256

# --- 1. Flatten Observation Wrapper for MLP ---
class FlattenObsWrapper(gym.ObservationWrapper):
    """
    Flattens a (H, W) board into a vector of size H*W for MLP input.
    Casts to float32 for compatibility with SB3.
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low.min(),
            high=env.observation_space.high.max(),
            shape=(BOARD_SIZE,),
            dtype=np.float32
        )

    def observation(self, obs):
        return obs.flatten().astype(np.float32)

# --- 2. Environment Setup ---
def make_minesweeper_env():
    return env_setup.MinesweeperEnv(height=HEIGHT, width=WIDTH, num_mines=NUM_MINES)

# Wrap environment before vectorization
sb3_env = make_vec_env(lambda: FlattenObsWrapper(make_minesweeper_env()), n_envs=1)

# --- 3. PPO Initialization ---
model = PPO(
    "MlpPolicy",  # Use MLP for flattened numeric board
    sb3_env,
    verbose=1,
)

# --- 4. Load DAgger Weights ---
POLICY_WEIGHTS_PATH = "minesweeper_dagger_policy_fixed.pth"
policy = model.policy

# Your MinesweeperPolicy is MLP, so loading weights should work with strict=False
state_dict_dagger = torch.load(POLICY_WEIGHTS_PATH)

try:
    policy.load_state_dict(state_dict_dagger, strict=False)
    print(f"✅ Successfully loaded DAgger weights into PPO Actor layers.")
except RuntimeError as e:
    print(f"❌ ERROR: Loading failed even with strict=False. Error: {e}")

# --- 5. Training ---
TOTAL_TIMESTEPS = 500000 # Adjust as needed for testing
print(f"\n🚀 Starting PPO fine-tuning for {TOTAL_TIMESTEPS} timesteps...")
model.learn(total_timesteps=TOTAL_TIMESTEPS)

# --- 6. Saving Model ---
MODEL_SAVE_NAME = "ppo_minesweeper_trained"
model.save(MODEL_SAVE_NAME)
print(f"💾 Full DAgger + PPO model saved as {MODEL_SAVE_NAME}.zip")

# Save only the policy weights
WEIGHTS_SAVE_NAME = "ppo_minesweeper_weights_only.pth"
torch.save(model.policy.state_dict(), WEIGHTS_SAVE_NAME)
print(f"💾 Policy weights saved as {WEIGHTS_SAVE_NAME}")
