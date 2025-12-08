from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, ProgressBarCallback
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
from FlatteningWrapper import FlattenObservationWrapper
import env_setup

height = 16
width = 16
board_size = height * width
num_mines = 40

def make_minesweeper_env():
    env = env_setup.MinesweeperEnv(height = height, width = width, num_mines = num_mines)
    return FlattenObservationWrapper(env)

sb3_env = make_vec_env(lambda: make_minesweeper_env(), n_envs = 1)
model = PPO('MlpPolicy', sb3_env, verbose = 1)

# Training
total_timesteps = 500000
print(f'Starting PPO training for {total_timesteps} timesteps...')
model.learn(total_timesteps = total_timesteps)

# Save Model
model_save_name = 'ppo_intermediate_500k'
model.save(model_save_name)
print(f'PPO training complete for baseline model, saved to {model_save_name}.zip')