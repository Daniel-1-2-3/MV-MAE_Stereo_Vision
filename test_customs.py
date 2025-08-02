import gymnasium as gym
import metaworld
import numpy as np
from SawyerSim.test_stereo_env import SawyerReachEnvV3

env = SawyerReachEnvV3(render_mode="human")
observation, info = env.reset()

for i in range(100000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(i, reward, info)
    if truncated:
        env.reset()
    env.render()
env.close()