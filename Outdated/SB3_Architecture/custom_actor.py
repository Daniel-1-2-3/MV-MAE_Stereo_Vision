from typing import Any
import numpy as np

import torch
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    create_mlp,
)

from stable_baselines3.common.type_aliases import PyTorchObs
from MAE_Model.model import MAEModel

# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20

class Actor(BasePolicy):
    """
    Actor network (policy) for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    action_space: spaces.Box

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: list[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = False,
    
        nviews: int = 2,
        mvmae_patch_size: int = 6, 
        mvmae_encoder_embed_dim: int = 768, 
        mvmae_decoder_embed_dim: int = 512,
        mvmae_encoder_heads: int = 16, 
        mvmae_decoder_heads: int = 16,
        masking_ratio: float = 0.75,
        in_channels: int = 3,
        img_h_size: int = 84,
        img_w_size: int = 84,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )
        
        # Initialize MVMAE, default params, small embed dims to save memory
        self.mvmae = MAEModel(
            nviews=nviews,
            patch_size=mvmae_patch_size,
            encoder_embed_dim=mvmae_encoder_embed_dim,
            decoder_embed_dim=mvmae_decoder_embed_dim,
            encoder_heads=mvmae_encoder_heads,
            decoder_heads=mvmae_decoder_heads,
            masking_ratio=masking_ratio,
            in_channels=in_channels,
            img_h_size=img_h_size,
            img_w_size=img_w_size
        ).to(self.device)
        
        # Save arguments to re-create object at loading
        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.net_arch = net_arch
        
        self.nviews = nviews 
        self.total_patches = (img_h_size // mvmae_patch_size) * (2 * img_w_size // mvmae_patch_size) # When fused view
        self.features_dim = self.total_patches * self.mvmae.encoder_embed_dim
        print("ACTOR FEATURES_DIM:", self.features_dim)
        
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean

        action_dim = get_action_dim(self.action_space)
        latent_pi_net = create_mlp(self.features_dim, -1, net_arch, activation_fn)
        self.latent_pi = nn.Sequential(*latent_pi_net)
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else self.features_dim
        
        if self.use_sde:
            self.action_dist = StateDependentNoiseDistribution(
                action_dim, full_std=full_std, use_expln=use_expln, learn_features=True, squash_output=True
            )
            self.mu, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=last_layer_dim, latent_sde_dim=last_layer_dim, log_std_init=log_std_init
            )
            # Avoid numerical issues by limiting the mean of the Gaussian
            # to be in [-clip_mean, clip_mean]
            if clip_mean > 0.0:
                self.mu = nn.Sequential(self.mu, nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean))
        else:
            self.action_dist = SquashedDiagGaussianDistribution(action_dim)  # type: ignore[assignment]
            self.mu = nn.Linear(last_layer_dim, action_dim)
            self.log_std = nn.Linear(last_layer_dim, action_dim)  # type: ignore[assignment]
            
        self.to(self.device)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                full_std=self.full_std,
                use_expln=self.use_expln,
                features_extractor=self.features_extractor,
                clip_mean=self.clip_mean,
            )
        )
        return data

    def get_std(self) -> torch.Tensor:
        """
        Retrieve the standard deviation of the action distribution.
        Only useful when using gSDE.
        It corresponds to ``torch.exp(log_std)`` in the normal case,
        but is slightly different when using ``expln`` function
        (cf StateDependentNoiseDistribution doc).

        :return:
        """
        msg = "get_std() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        return self.action_dist.get_std(self.log_std)

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.

        :param batch_size:
        """
        msg = "reset_noise() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        self.action_dist.sample_weights(self.log_std, batch_size=batch_size)
        
    # OVERRIDE the extract_features(): process obs using mvmae encoder
    def extract_features(self, obs: PyTorchObs) -> torch.Tensor:
        """
            Observation is a Tensor or np.ndarray, shape (B, H, 2W, C), float32 (standardized).
        """
        # Ensure image is float32 and on correct device
        img = obs
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img).float()
        else:
            if img.dtype != torch.float32:
                img = img.to(dtype=torch.float32)

        # Move to GPU with pinned memory + non blocking if available
        if self.device.type == "cuda":
            if img.device.type == "cpu":
                img = img.pin_memory().to(self.device, non_blocking=True)
            elif img.device != self.device:
                img = img.to(self.device, non_blocking=True)
        else:
            if img.device != self.device:
                img = img.to(self.device)
        
        z, mask = self.mvmae.encoder(img, mask_x=False)
        return z.flatten(start_dim=-2), img
        
    def get_action_dist_params(self, obs: PyTorchObs) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """
        Get the parameters for the action distribution.

        :param obs:
        :return:
            Mean, standard deviation and optional keyword arguments.
        """
        z, img = self.extract_features(obs)
        latent_pi = self.latent_pi(z)
        mean_actions = self.mu(latent_pi)

        if self.use_sde:
            return mean_actions, self.log_std, dict(latent_sde=latent_pi)
        # Unstructured exploration (Original implementation)
        log_std = self.log_std(latent_pi)  # type: ignore[operator]
        # Original Implementation to cap the standard deviation
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> torch.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # Note: the action is squashed
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)

    def action_log_prob(self, obs: PyTorchObs) -> tuple[torch.Tensor, torch.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # return action and associated log prob
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)
    
    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> torch.Tensor:
        obs_tensor = torch.from_numpy(observation).float() if not isinstance(observation, torch.Tensor) else observation.to(dtype=torch.float32)
        obs_tensor = obs_tensor.to(self.device, non_blocking=True)
        return self(obs_tensor, deterministic)