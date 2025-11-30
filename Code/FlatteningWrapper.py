import gymnasium as gym
import numpy as np

class FlattenObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Update the observation space to reflect flattened shape
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low.flatten().astype(np.float32),
            high=env.observation_space.high.flatten().astype(np.float32),
            dtype=np.float32
        )

    def observation(self, obs):
        return obs.flatten().astype(np.float32)
