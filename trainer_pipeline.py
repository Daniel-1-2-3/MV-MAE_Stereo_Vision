import os
os.environ["MUJOCO_GL"] = "osmesa"

from SawyerSim.sawyer_stereo_env import SawyerReachEnvV3
from SawyerSim.custom_sac import SAC
from SawyerSim.custom_sac_policy import SACPolicy

env = SawyerReachEnvV3(render_mode="rgb_array")
model = SAC(SACPolicy, env, buffer_size=500, verbose=1, batch_size=32)
model.learn(total_timesteps=10_000, log_interval=4, progress_bar=True)