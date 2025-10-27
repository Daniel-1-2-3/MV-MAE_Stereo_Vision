"""Policies: abstract base class and concrete implementations."""

import torch
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.policies import BaseModel
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    create_mlp,
)
from stable_baselines3.common.type_aliases import PyTorchObs
from MAE_Model.model import MAEModel

class ContinuousCritic(BaseModel):
    """
    Critic network(s) for DDPG/SAC/TD3.
    It represents the action-state value function (Q-value function).
    Compared to A2C/PPO critics, this one represents the Q-value
    and takes the continuous action as input. It is concatenated with the state
    and then fed to the network which outputs a single value: Q(s, a).
    For more recent algorithms like SAC/TD3, multiple networks
    are created to give different estimates.

    By default, it creates two critic networks used to reduce overestimation
    thanks to clipped Q-learning (cf TD3 paper).

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether the features extractor is shared or not
        between the actor and the critic (this saves computation time)
    """

    features_extractor: BaseFeaturesExtractor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: list[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = False,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        self.action_dim = get_action_dim(self.action_space)
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.features_dim = None
        self.q_networks: list[nn.Module] = []
        
        self.encoder = None
    
    def make_q_networks(self):
        for idx in range(self.n_critics):
            q_net_list = create_mlp(self.features_dim + self.action_dim, 1, self.net_arch, self.activation_fn)
            q_net = nn.Sequential(*q_net_list)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)
            
    #OVERRIDE extract_features, take the "state" in the dictionary that is my current observation
    def extract_features(self, obs) -> torch.Tensor:
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
        with torch.no_grad():
            z, mask = self.encoder(img, mask_x=False)
        return z.flatten(start_dim=-2)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        features = self.extract_features(obs)
        qvalue_input = torch.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def set_encoder(self, encoder):
        self.encoder = encoder

    def q1_forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with torch.no_grad():
            z = self.extract_features(obs)
        return self.q_networks[0](torch.cat([z, actions], dim=1))
