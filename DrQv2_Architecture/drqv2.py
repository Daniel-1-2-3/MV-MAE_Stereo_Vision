import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import DrQv2_Architecture.utils as utils
from MAE_Model.model import MAEModel
from MAE_Model.encoder import ViTMaskedEncoder
from gymnasium.spaces import Box

class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.LayerNorm(feature_dim), nn.Tanh()
        )

        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_shape[0])
        )

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist

class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2
class DrQV2Agent:
    def __init__(self,
        # General variables
        action_shape: tuple | None = None,
        device: torch.device | None = None,
        lr: float = 1e-4,
        # MVMAE variables
        nviews: int = 2,
        mvmae_patch_size: int = 8, 
        mvmae_encoder_embed_dim: int = 256, 
        mvmae_decoder_embed_dim: int = 128,
        mvmae_encoder_heads: int = 16,
        in_channels: int = 9,
        img_h_size: int = 64,
        img_w_size: int = 64,
        mvmae_decoder_heads: int = 16,
        masking_ratio: float = 0.75,
        coef_mvmae: float = 0.005,
        # Actor
        feature_dim: int = 100,
        hidden_dim: int = 1024,
        # RL variables
        critic_target_tau: float = 0.001, # Soft-update for target critic
        num_expl_steps: int = 2000, 
        update_every_steps: int = 2,
        stddev_schedule: str = 'linear(1.0,0.1,500000)', # Type of scheduler, value taken from cfgs/task/medium.yaml, stddev for exploration noise
        stddev_clip: int = 0.3, # How much to clip sampled action noise
        use_tb: bool = True,
    ):
        self.action_shape = action_shape
        self.device = device
        self.lr = lr

        self.nviews = nviews
        self.mvmae_patch_size = mvmae_patch_size
        self.mvmae_encoder_embed_dim = mvmae_encoder_embed_dim
        self.mvmae_decoder_embed_dim = mvmae_decoder_embed_dim
        self.mvmae_encoder_heads = mvmae_encoder_heads
        self.mvmae_decoder_heads = mvmae_decoder_heads
        self.in_channels = in_channels
        self.img_h_size = img_h_size
        self.img_w_size = img_w_size
        self.masking_ratio = masking_ratio
        self.coef_mvmae = coef_mvmae
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        self.critic_target_tau = critic_target_tau
        self.num_expl_steps = num_expl_steps
        self.update_every_steps = update_every_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.use_tb = use_tb
        
        # Models
        self.mvmae = MAEModel(
            nviews=self.nviews,
            patch_size=self.mvmae_patch_size,
            encoder_embed_dim=self.mvmae_encoder_embed_dim,
            decoder_embed_dim=self.mvmae_decoder_embed_dim,
            encoder_heads=self.mvmae_encoder_heads,
            decoder_heads=mvmae_decoder_heads,
            in_channels=self.in_channels,
            img_h_size=self.img_h_size,
            img_w_size=self.img_w_size,
            masking_ratio=self.masking_ratio
        ).to(self.device)
        
        # The dimension of the flattened encoder output z
        self.total_patches = (img_h_size // mvmae_patch_size) * (2 * img_w_size // mvmae_patch_size) # When fused view
        self.repr_dim = self.total_patches * self.mvmae.encoder_embed_dim
        
        self.actor = Actor(self.repr_dim, action_shape, self.feature_dim, self.hidden_dim).to(device)
        self.critic = Critic(self.repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.critic_target = Critic(self.repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.mvmae_optim = torch.optim.Adam(self.mvmae.parameters(), lr=lr)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.train()
        self.critic_target.train()

    # Set into training (train vs eval) mode
    def train(self, training=True):
        self.training = training
        self.mvmae.train(training)
        self.actor.train(training)
        self.critic.train(training)

    # Samples an action
    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            z, _ = self.mvmae.encoder(obs.unsqueeze(0), mask_x=False)
            obs_latent = z.flatten(start_dim=-2)
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(obs_latent, stddev)
            if eval_mode:
                action = dist.mean
            else:
                action = dist.sample(clip=None)
                if step < self.num_expl_steps:
                    action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, z, action, reward, discount, z_next, step, obs):
        metrics = dict()

        # Calculates target, computes MSE critic loss
        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(z_next, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(z_next, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(z, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
        
        out, mask, _ = self.mvmae.forward(obs, mask_x=True)
        recon_loss = self.mvmae.compute_loss(out, obs, mask)
        
        total_loss = critic_loss + self.coef_mvmae * recon_loss

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()
            metrics['recon_loss'] = recon_loss.item()
            metrics['total_loss'] = total_loss.item()

        # Backpropagate the critic loss + mvmae recon loss, together
        self.critic_optim.zero_grad(set_to_none=True)
        self.mvmae_optim.zero_grad(set_to_none=True)
        total_loss.backward()
        self.critic_optim.step()
        self.mvmae_optim.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # Optimize actor
        self.actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optim.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        # Sample a batch from the replay buffer
        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)
        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # Augment
        obs = obs.float()
        next_obs = next_obs.float()
        
        # Encoder used to process obs with gradients on, and next_obs with gradients off. 
        z: torch.Tensor
        z_next: torch.Tensor
        z, _ = self.mvmae.encoder(obs, mask_x=False)
        z = z.flatten(start_dim=-2)
        with torch.no_grad():
            z_next = self.mvmae.encoder(next_obs, mask_x=False)
            z_next = z_next.flatten(start_dim=-2)

        # Actor, critic, and encoder updates
        # Encoder reconstruction loss is calculated and backpropagated inside .update_critic()
        metrics.update(self.update_critic(z, action, reward, discount, z_next, step, obs))
        metrics.update(self.update_actor(z.detach(), step))

        # Update critic target
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

        return metrics
