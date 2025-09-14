# Only when running in RunPod hosted GPU, not local environment
import os
os.environ["MUJOCO_GL"] = "osmesa"

from SawyerSim.sawyer_stereo_env import SawyerReachEnvV3
from SawyerSim.custom_sac import Custom_SAC
from SawyerSim.custom_sac_policy import SACPolicy
import numpy as np
import argparse

class PipelineTrainer():
    def __init__(
        self,
        render_mode: str = "human",
        learning_rate = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 50_000,
        batch_size: int = 256,
        gamma: float = 0.99,
        n_steps: int = 1, # Number of steps before update
        ent_coef = "auto",
        target_entropy = "auto",
        verbose: int = 0,
        nviews: int = 2,
        mvmae_patch_size: int = 6, 
        mvmae_encoder_embed_dim: int = 768, 
        mvmae_decoder_embed_dim: int = 512,
        mvmae_encoder_heads: int = 16, 
        mvmae_decoder_heads: int = 16,
        in_channels: int = 3,
        img_h_size: int = 84,
        img_w_size: int = 84,
        total_timesteps: int = 5_000_000,
        episode_horizon: int = 300,
    ):
        self.env = SawyerReachEnvV3(
            render_mode = render_mode,
            img_width = img_w_size,
            img_height = img_h_size,
            max_path_length = episode_horizon
        )

        self.model = Custom_SAC(
            SACPolicy,
            self.env,
            learning_rate = learning_rate,
            buffer_size = buffer_size,
            learning_starts = learning_starts,
            batch_size = batch_size,
            gamma = gamma,
            n_steps = n_steps,
            ent_coef = ent_coef,
            target_entropy = target_entropy,
            verbose = verbose,
            policy_kwargs= {
                "nviews" : nviews,
                "mvmae_patch_size" : mvmae_patch_size, 
                "mvmae_encoder_embed_dim" : mvmae_encoder_embed_dim, 
                "mvmae_decoder_embed_dim" : mvmae_decoder_embed_dim,
                "mvmae_encoder_heads" : mvmae_encoder_heads, 
                "mvmae_decoder_heads" : mvmae_decoder_heads,
                "in_channels" : in_channels,
            }
        )
        
        self.total_timesteps = total_timesteps
    
    def train(self):
        self.model.begin_log_losses()
        self.model.learn(total_timesteps=self.total_timesteps, log_interval=4, progress_bar=True)
        self.model.save("metaworld_sac_mvmae")
        
    def eval(self):
        del self.model
        self.model = Custom_SAC.load("metaworld_sac_mvmae")

        obs, info = self.env.reset()
        for i in range(0, 100_000):
            action, _states = self.model.predict(obs, deterministic=True)
            action = np.squeeze(action) 
            obs, reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                obs, info = self.env.reset()

def get_args():
    parser = argparse.ArgumentParser(description="Trainer Pipeline Config")

    # RL hyperparameters
    parser.add_argument("--render_mode", type=str, default="rgb_array")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--buffer_size", type=int, default=1_000_000)
    parser.add_argument("--learning_starts", type=int, default=50_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n_steps", type=int, default=1,
                        help="Number of steps before update")
    parser.add_argument("--ent_coef", type=str, default="auto",
                        help="Entropy coefficient (float or 'auto')")
    parser.add_argument("--target_entropy", type=str, default="auto")
    parser.add_argument("--verbose", type=int, default=0)

    # MV-MAE hyperparameters
    parser.add_argument("--nviews", type=int, default=2)
    parser.add_argument("--mvmae_patch_size", type=int, default=6)
    parser.add_argument("--mvmae_encoder_embed_dim", type=int, default=768)
    parser.add_argument("--mvmae_decoder_embed_dim", type=int, default=512)
    parser.add_argument("--mvmae_encoder_heads", type=int, default=16)
    parser.add_argument("--mvmae_decoder_heads", type=int, default=16)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--img_h_size", type=int, default=84)
    parser.add_argument("--img_w_size", type=int, default=84)
    
    parser.add_argument("--total_timesteps", type=int, default=5_000_000)
    parser.add_argument("--episode_horizon", type=int, default=300)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    
    warn_msg = "Image should have dimensions so that it is divisible, without remainder, into patches. Change either the image dims or patch size."
    assert (args.img_h_size * args.img_w_size) % args.mvmae_patch_size == 0, warn_msg
    
    trainer = PipelineTrainer(**vars(args))
    trainer.train()
    trainer.eval()