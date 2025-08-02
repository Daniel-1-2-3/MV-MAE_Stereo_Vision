import gymnasium as gym
import numpy as np
from gymnasium import spaces
from metaworld.sawyer_xyz_env import SawyerXYZEnv

class StereoEnv(SawyerXYZEnv):
    metadata= {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        render_mode="human",
        camera_name: str | None = None,
        camera_id: int | None = None,
        reward_function_version: str = "v2",
        height: int = 480,
        width: int = 480,
    ):
        super().__init__()
        
        self.action_space = None
        self.observation_space = None
        
    def step(self, action):
        observation, reward, terminated, truncated, info = [None] * 5
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        observation, info = [None] * 2
        return observation, info
    
    def render(self):
        pass
    
    def close(self):
        pass