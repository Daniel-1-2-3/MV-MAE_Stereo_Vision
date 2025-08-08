# Only when running in RunPod hosted GPU, not local environment
# import os
# os.environ["MUJOCO_GL"] = "osmesa"

from SawyerSim.sawyer_stereo_env import SawyerReachEnvV3
from SawyerSim.custom_sac import SAC
from SawyerSim.custom_sac_policy import SACPolicy

env = SawyerReachEnvV3(render_mode="rgb_array") # or "human" for rendering
model = SAC(SACPolicy, env, buffer_size=1_000_000, verbose=1, batch_size=128, learning_starts=50_000)
model.learn(total_timesteps=1, log_interval=4, progress_bar=True)
model.save("metaworld_sac_mvmae")


del model
model = SAC.load("metaworld_sac_mvmae")

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()