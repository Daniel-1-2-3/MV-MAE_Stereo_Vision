import gymnasium as gym
import numpy as np
from SawyerSim.stereo_env import SawyerReachEnvV3

env = SawyerReachEnvV3(render_mode="human")
observation, info = env.reset()

for i in range(2000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if truncated:
        env.reset()