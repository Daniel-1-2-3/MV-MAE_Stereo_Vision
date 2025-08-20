# Only when running in RunPod hosted GPU, not local environment
# import os
# os.environ["MUJOCO_GL"] = "osmesa"

from SawyerSim.sawyer_stereo_env import SawyerReachEnvV3
from SawyerSim.custom_sac import Custom_SAC
from SawyerSim.custom_sac_policy import SACPolicy
import numpy as np

env = SawyerReachEnvV3(render_mode="human", img_width=84, img_height=84) # or "human" for rendering
# Buffer 1000000, batch_size 128, learningstarts 50000
model = Custom_SAC(SACPolicy, env, buffer_size=100, verbose=1, batch_size=32, learning_starts=10, 
                   policy_kwargs={"nviews": 2,
                                  "mvmae_patch_size": 8, 
                                  "mvmae_encoder_embed_dim": 768, 
                                  "mvmae_decoder_embed_dim": 512,
                                  "mvmae_encoder_heads": 16, 
                                  "mvmae_decoder_heads": 16,
                                  "in_channels": 3,
                                  "img_h_size": 80,
                                  "img_w_size": 80,
                                })
model.begin_log_losses()
model.learn(total_timesteps=5_000_000, log_interval=4, progress_bar=True)
model.save("metaworld_sac_mvmae")

del model
model = Custom_SAC.load("metaworld_sac_mvmae")

obs, info = env.reset()
for i in range(0, 100_000):
    action, _states = model.predict(obs, deterministic=True)
    action = np.squeeze(action) 
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()