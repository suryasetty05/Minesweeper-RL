# PPO (Policy Proximal Optimization) implementation with MLP for Minesweeper RL
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
from FlatteningWrapper import FlattenObservationWrapper
from env_setup import create_minesweeper_env

# 1) Create and wrap the environment
def make_wrapped_env(**kwargs):
    env = create_minesweeper_env(**kwargs)
    env = FlattenObservationWrapper(env)  # flatten observations
    return env

# Create vectorized environments 
env = make_vec_env(
    make_wrapped_env,
    n_envs=8,
    env_kwargs=dict(
        height=8,
        width=8,
        num_mines = 10
    )
)

policy_kwargs = dict(
    net_arch=[256, 256]  # Two hidden layers with 256 units each
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
    verbose=1,
    device='auto'
)

# 4) Callbacks for eval / checkpoint
eval_env = make_wrapped_env(
    height=8,
    width=8,
    num_mines = 10
)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/best_model/",
    log_path="./logs/results/",
    eval_freq=5000,         # evaluate every 5k timesteps
    n_eval_episodes=50,
    deterministic=True,
)

checkpoint_callback = CheckpointCallback(
    save_freq=50000,        # checkpoint every 50k timesteps
    save_path='./logs/checkpoints/',
    name_prefix='ppo_minesweeper'
)

# 5) Train
total_timesteps = 500_000
model.learn(total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback])

# 6) Save final
model.save("ppo_minesweeper_500k")

