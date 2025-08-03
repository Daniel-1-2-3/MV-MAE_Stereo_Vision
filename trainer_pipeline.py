import gymnasium as gym
import numpy as np
from SawyerSim.stereo_env import SawyerReachEnvV3
from SawyerSim.sac_mae_policy import SAC_MAE

env = SawyerReachEnvV3(render_mode="human")
model = SAC_MAE(env).learn(total_timesteps=1000)