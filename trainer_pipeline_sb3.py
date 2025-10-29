# Only when running in RunPod hosted GPU, not local environment
#import os
#os.environ.setdefault("MUJOCO_GL", "osmesa")
#os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

from SawyerSim.sawyer_stereo_env import SawyerReachEnvV3
from SB3_Architecture.custom_sac import Custom_SAC
from SB3_Architecture.custom_sac_policy import SACPolicy
from SB3_Architecture.debugger import Debugger
import numpy as np
import argparse
import torch

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
        masking_ratio: float = 0.75,
        coef_mvmae: float = 0.1,
        in_channels: int = 3,
        img_h_size: int = 84,
        img_w_size: int = 84,
        total_timesteps: int = 5_000_000,
        episode_horizon: int = 300,
        log_file: str = 'log.csv',
    ):  
        self.debugger = Debugger()
        
        self.env = SawyerReachEnvV3(
            debugger = self.debugger,
            render_mode = render_mode,
            img_width = img_w_size,
            img_height = img_h_size,
            max_path_length = episode_horizon
        )

        self.model = Custom_SAC(
            SACPolicy,
            self.env,
            device="cuda" if torch.cuda.is_available() else "cpu",
            learning_rate = learning_rate,
            buffer_size = buffer_size,
            learning_starts = learning_starts,
            batch_size = batch_size,
            gamma = gamma,
            n_steps = n_steps,
            ent_coef = ent_coef,
            target_entropy = target_entropy,
            verbose = verbose,
            coef_mvmae = coef_mvmae,
            log_file=log_file,
            debugger = self.debugger,
            policy_kwargs= {
                "nviews" : nviews,
                "mvmae_patch_size" : mvmae_patch_size, 
                "mvmae_encoder_embed_dim" : mvmae_encoder_embed_dim, 
                "mvmae_decoder_embed_dim" : mvmae_decoder_embed_dim,
                "mvmae_encoder_heads" : mvmae_encoder_heads, 
                "mvmae_decoder_heads" : mvmae_decoder_heads,
                "img_h_size": img_h_size,
                "img_w_size": img_w_size,
                "in_channels" : in_channels,
                "masking_ratio": masking_ratio,
            }
        )
        # Assuming self.model is your Custom_SAC instance
        self.print_model_devices(self.model.policy, name="SAC Policy")

        # If you want to drill into your custom actor/MAE
        self.print_model_devices(self.model.policy.actor, name="Actor Network")
        self.print_model_devices(self.model.policy.critic, name="Critic Network")
        
        self.total_timesteps = total_timesteps
    
    def train(self):
        self.model.begin_log_losses()
        self.model.learn(total_timesteps=self.total_timesteps, log_interval=4, progress_bar=False)
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
    
    def print_model_devices(self, model, name="Model"):
        """Print which device each submodule is on."""
        print(f"\n=== Device report for {name} ===")
        for module_name, module in model.named_modules():
            if module_name == "":
                continue
            try:
                p = next(module.parameters())
                print(f"{module_name:<40} -> {p.device}")
            except StopIteration:
                continue
        print("================================\n")

def get_args():
    parser = argparse.ArgumentParser(description="Trainer Pipeline Config")

    # RL hyperparameters
    parser.add_argument("--render_mode", type=str, default="rgb_array")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--buffer_size", type=int, default=100_000)
    parser.add_argument("--learning_starts", type=int, default=50_000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n_steps", type=int, default=1,
                        help="Number of steps before update")
    parser.add_argument("--ent_coef", type=str, default="auto",
                        help="Entropy coefficient (float or 'auto')")
    parser.add_argument("--target_entropy", type=str, default="auto")
    parser.add_argument("--verbose", type=int, default=0)

    # MV-MAE hyperparameters
    parser.add_argument("--nviews", type=int, default=2)
    parser.add_argument("--mvmae_patch_size", type=int, default=8)
    parser.add_argument("--mvmae_encoder_embed_dim", type=int, default=256)
    parser.add_argument("--mvmae_decoder_embed_dim", type=int, default=128)
    parser.add_argument("--mvmae_encoder_heads", type=int, default=16)
    parser.add_argument("--mvmae_decoder_heads", type=int, default=16)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--img_h_size", type=int, default=64)
    parser.add_argument("--img_w_size", type=int, default=64)
    
    parser.add_argument("--total_timesteps", type=int, default=5_000_000)
    parser.add_argument("--episode_horizon", type=int, default=300)
    parser.add_argument("--coef_mvmae", type=float, default=0.1)
    parser.add_argument("--masking_ratio", type=float, default=0.75)
    
    parser.add_argument("--log_file", type=str, default='log.csv')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    
    assert args.img_h_size % args.mvmae_patch_size == 0 and args.img_w_size % args.mvmae_patch_size == 0, \
    "Height and width must be divisible by patch size."

    trainer = PipelineTrainer(**vars(args))
    
    trainer.train()
    trainer.eval()