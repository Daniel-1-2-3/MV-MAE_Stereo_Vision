import os
os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import torch
import cv2
import mujoco
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import get_schedule_fn

from FrankaSim.franka_env import FrankaEnv
from FrankaSim.mvmae_feature_extractor import MVMAEFeatureExtractor
from FrankaSim.custom_policy import CustomSAC
from stable_baselines3.sac import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
import argparse

# Custom Policy with MV-MAE feature extractor
class CustomSACPolicy(MlpPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=MVMAEFeatureExtractor,
            features_extractor_kwargs={
                "nviews": 2,
                "patch_size": 8,
                "encoder_embed_dim": 768,
                "decoder_embed_dim": 512,
                "encoder_heads": 16,
                "decoder_heads": 16,
                "in_channels": 3,
                "img_h_size": 128,
                "img_w_size": 256,
            },
            **kwargs,
        )

# Render frames after each environment step
class RenderCallback(BaseCallback):
    def __init__(self, env: FrankaEnv, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.renderer = mujoco.Renderer(env.model)
        self.left_cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "left_eye")
        self.right_cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "right_eye")

    def _on_step(self) -> bool:
        self.renderer.update_scene(self.env.data, camera=self.left_cam_id)
        left_img = cv2.resize(self.renderer.render(), dsize=None, fx=2.0, fy=2.0)

        self.renderer.update_scene(self.env.data, camera=self.right_cam_id)
        right_img = cv2.resize(self.renderer.render(), dsize=None, fx=2.0, fy=2.0)

        stereo_view = np.concatenate((left_img, right_img), axis=1)
        stereo_view_bgr = cv2.cvtColor(stereo_view, cv2.COLOR_RGB2BGR)

        cv2.imshow("Stereo View (Left | Right)", stereo_view_bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False  # Stop training
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=4)
    args = parser.parse_args()
    
    def make_franka_env():
        return FrankaEnv(
            model_path=os.path.join(os.getcwd(), "FrankaSim", "pick_place.xml"),
            render_mode="rgb_array",
            n_substeps=25,
            reward_type="dense",
            distance_threshold=0.05,
            goal_xy_range=0.3,
            obj_xy_range=0.3,
            goal_x_offset=0.0,
            goal_z_range=0.2)
    env = make_vec_env(make_franka_env, n_envs=args.envs, vec_env_cls=DummyVecEnv)
    env.reset() 
    
    model = CustomSAC(
        policy=CustomSACPolicy,
        env=env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=50_000,
        batch_size=32,
        learning_starts=50, # Only starts training after some buffer has been filled, use 5000 for actual training
        train_freq = (16, "step"),
        gradient_steps = 32,
        gamma=0.99,
        tau=0.005,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    try:
        model.learn(
            total_timesteps=100_000,
            # callback=RenderCallback(env), 
            log_interval=10,
            progress_bar=True
        )
    finally:
        env.close()
        cv2.destroyAllWindows()