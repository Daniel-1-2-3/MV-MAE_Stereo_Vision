import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict
from Model.model import Model
import gym

class MVMAEFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, 
        observation_space: gym.spaces.Dict,
        nviews=2,
        patch_size=8,
        encoder_embed_dim=768,
        decoder_embed_dim=512,
        encoder_heads=16,
        decoder_heads=16,
        in_channels=3,
        img_h_size=128,
        img_w_size=128, 
    ):
        super().__init__(observation_space, features_dim=encoder_embed_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mvmae = Model(
            nviews=nviews,
            patch_size=patch_size,
            encoder_embed_dim=encoder_embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            encoder_heads=encoder_heads,
            decoder_heads=decoder_heads,
            in_channels=in_channels,
            img_h_size=img_h_size,
            img_w_size=img_w_size
        )
        self.mvmae.to(self.device)
        self.last_mvmae_loss = None

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        obs = obs_dict["observation"].float().to(self.device)
        # obs is (batch, height, width * 2, channels)
        self.mvmae.train()
        out, mask, encoder_nomask_x = self.mvmae(obs)
        
        # Get the mvmae loss
        self.last_mvmae_loss = self.mvmae.compute_loss(out, obs, mask) * 2
        
        # See model .forward() for description of out, mask, and encoder_nomask_x
        pooled_features = encoder_nomask_x.mean(dim=1)
        return pooled_features # Return features for SAC policy (unmasked encoder output)
