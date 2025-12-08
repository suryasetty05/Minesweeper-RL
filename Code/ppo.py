# PPO (Policy Proximal Optimization) implementation with MLP for Minesweeper RL
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, ProgressBarCallback
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
from FlatteningWrapper import FlattenObservationWrapper
from env_setup import create_minesweeper_env

def make_wrapped_env(**kwargs):
    env = create_minesweeper_env(**kwargs)
    env = FlattenObservationWrapper(env)
    return env

ENV_CONFIG = dict(
    height=16,
    width=16,
    num_mines = 40
)

env = make_vec_env(
    make_wrapped_env,
    n_envs=16,
    env_kwargs=ENV_CONFIG
)

policy_kwargs = dict(
    net_arch=[512, 256]
)

model = PPO(
    "MlpPolicy",
    env = env,
    policy_kwargs=policy_kwargs,
    learning_rate=2.5e-4,
    n_steps=512,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=1e-4,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=0,
    device='auto'
)

eval_env = make_wrapped_env(**ENV_CONFIG)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/model2/",
    log_path="./logs/results",
    eval_freq=5000,
    n_eval_episodes=50,
    deterministic=True,
)

checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path='./logs/checkpoints/',
    name_prefix='ppo_minesweeper2'
)

progress_bar_callback = ProgressBarCallback()

total_timesteps = 500_000
print(f"Starting PPO training for {total_timesteps} timesteps on 8x8 board...")
model.learn(total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback, progress_bar_callback])

model.save("ppo_intermediate_500k")
print("PPO training complete and final model saved.")