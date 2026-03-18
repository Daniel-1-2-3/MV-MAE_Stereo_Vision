import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import DrQv2_Architecture.utils as utils
from MAE_Model.model import MAEModel
from MAE_Model.encoder import ViTMaskedEncoder
from gymnasium.spaces import Box

import time

def _cuda_sync(device):
    if device is not None and "cuda" in str(device) and torch.cuda.is_available():
        torch.cuda.synchronize(device)

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
        action_shape: tuple | None = None,
        device: torch.device | None = None,
        lr: float = 1e-4,
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
        feature_dim: int = 100,
        hidden_dim: int = 1024,
        critic_target_tau: float = 0.001,
        num_expl_steps: int = 2000,
        update_every_steps: int = 2,
        update_mvmae_every_steps: int = 10,
        stddev_schedule: str = 'linear(1.0,0.1,500000)',
        stddev_clip: int = 0.3,
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
        self.update_mvmae_every_steps = update_mvmae_every_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.use_tb = use_tb

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

        self.total_patches = (img_h_size // mvmae_patch_size) * (2 * img_w_size // mvmae_patch_size)
        self.repr_dim = self.total_patches * self.mvmae.encoder_embed_dim

        self.actor = Actor(self.repr_dim, action_shape, self.feature_dim, self.hidden_dim).to(device)
        self.critic = Critic(self.repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.critic_target = Critic(self.repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.mvmae_optim = torch.optim.Adam(self.mvmae.parameters(), lr=lr)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.mvmae.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        timings = {}
        _cuda_sync(self.device)
        t0 = time.perf_counter()

        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)

        _cuda_sync(self.device)
        t1 = time.perf_counter()

        with torch.no_grad():
            z, _ = self.mvmae.encoder(obs.unsqueeze(0), mask_x=False)
            obs_latent = z.flatten(start_dim=-2)

        _cuda_sync(self.device)
        t2 = time.perf_counter()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs_latent, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)

        _cuda_sync(self.device)
        t3 = time.perf_counter()

        action_np = action.cpu().numpy()[0]

        _cuda_sync(self.device)
        t4 = time.perf_counter()

        timings["act_tensorize_ms"] = 1000.0 * (t1 - t0)
        timings["act_encoder_ms"]   = 1000.0 * (t2 - t1)
        timings["act_actor_ms"]     = 1000.0 * (t3 - t2)
        timings["act_to_cpu_ms"]    = 1000.0 * (t4 - t3)
        self.last_act_timings = timings

        return action_np

    def update_critic(self, z, action, reward, discount, z_next, step, obs, update_mvmae: bool):
        metrics = dict()

        # --- Target Q ---
        _cuda_sync(self.device)
        t0 = time.perf_counter()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(z_next, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(z_next, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        _cuda_sync(self.device)
        t1 = time.perf_counter()

        # --- Critic forward + loss ---
        Q1, Q2 = self.critic(z, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        _cuda_sync(self.device)
        t2 = time.perf_counter()

        # --- MAE forward + loss (conditional) ---
        recon_loss = None
        if update_mvmae:
            out, mask, _ = self.mvmae.forward(obs, mask_x=True)
            _cuda_sync(self.device)
            t_mae_fwd = time.perf_counter()

            recon_loss = self.mvmae.compute_loss(out, obs, mask)
            _cuda_sync(self.device)
            t_mae_loss = time.perf_counter()

            metrics['perf_critic_mae_forward_ms'] = 1000.0 * (t_mae_fwd - t2)
            metrics['perf_critic_mae_loss_ms']    = 1000.0 * (t_mae_loss - t_mae_fwd)
        else:
            t_mae_loss = t2

        # --- Combine losses ---
        total_loss = critic_loss
        if update_mvmae:
            total_loss = total_loss + self.coef_mvmae * recon_loss

        _cuda_sync(self.device)
        t3 = time.perf_counter()

        # --- Zero grads ---
        self.critic_optim.zero_grad(set_to_none=True)
        if update_mvmae:
            self.mvmae_optim.zero_grad(set_to_none=True)

        _cuda_sync(self.device)
        t4 = time.perf_counter()

        # --- Backward ---
        total_loss.backward()

        _cuda_sync(self.device)
        t5 = time.perf_counter()

        # --- Optimizer step ---
        self.critic_optim.step()
        if update_mvmae:
            self.mvmae_optim.step()

        _cuda_sync(self.device)
        t6 = time.perf_counter()

        # --- Soft update target critic ---
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

        _cuda_sync(self.device)
        t7 = time.perf_counter()

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1']       = Q1.mean().item()
            metrics['critic_q2']       = Q2.mean().item()
            metrics['critic_loss']     = critic_loss.item()
            metrics['recon_loss']      = recon_loss.item() if update_mvmae else -1.0
            metrics['total_loss']      = total_loss.item()

        metrics['perf_critic_target_q_ms']     = 1000.0 * (t1 - t0)
        metrics['perf_critic_forward_loss_ms'] = 1000.0 * (t2 - t1)
        metrics['perf_critic_zero_grad_ms']    = 1000.0 * (t4 - t3)
        metrics['perf_critic_backward_ms']     = 1000.0 * (t5 - t4)
        metrics['perf_critic_optim_step_ms']   = 1000.0 * (t6 - t5)
        metrics['perf_critic_soft_update_ms']  = 1000.0 * (t7 - t6)
        metrics['perf_update_critic_total_ms'] = 1000.0 * (t7 - t0)

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        _cuda_sync(self.device)
        t0 = time.perf_counter()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)
        actor_loss = -Q.mean()

        _cuda_sync(self.device)
        t1 = time.perf_counter()

        self.actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optim.step()

        _cuda_sync(self.device)
        t2 = time.perf_counter()

        if self.use_tb:
            metrics['actor_loss']    = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent']     = dist.entropy().sum(dim=-1).mean().item()

        metrics['perf_actor_forward_ms']       = 1000.0 * (t1 - t0)
        metrics['perf_actor_backward_step_ms'] = 1000.0 * (t2 - t1)
        metrics['perf_update_actor_total_ms']  = 1000.0 * (t2 - t0)

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics
        update_mvmae = step % self.update_mvmae_every_steps == 0

        _cuda_sync(self.device)
        t0 = time.perf_counter()

        batch = next(replay_iter)

        _cuda_sync(self.device)
        t1 = time.perf_counter()

        obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)

        _cuda_sync(self.device)
        t2 = time.perf_counter()

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        obs = obs.float()
        next_obs = next_obs.float()

        z, _ = self.mvmae.encoder(obs, mask_x=False)
        z = z.flatten(start_dim=-2)

        _cuda_sync(self.device)
        t3 = time.perf_counter()

        with torch.no_grad():
            z_next, _ = self.mvmae.encoder(next_obs, mask_x=False)
            z_next = z_next.flatten(start_dim=-2)

        _cuda_sync(self.device)
        t4 = time.perf_counter()

        metrics_c = self.update_critic(z, action, reward, discount, z_next, step, obs, update_mvmae)

        _cuda_sync(self.device)
        t5 = time.perf_counter()

        metrics_a = self.update_actor(z.detach(), step)

        _cuda_sync(self.device)
        t6 = time.perf_counter()

        metrics.update(metrics_c)
        metrics.update(metrics_a)

        metrics['perf_replay_next_ms']         = 1000.0 * (t1 - t0)
        metrics['perf_to_torch_ms']            = 1000.0 * (t2 - t1)
        metrics['perf_encode_obs_ms']          = 1000.0 * (t3 - t2)
        metrics['perf_encode_next_obs_ms']     = 1000.0 * (t4 - t3)
        metrics['perf_update_critic_total_ms'] = 1000.0 * (t5 - t4)
        metrics['perf_update_actor_total_ms']  = 1000.0 * (t6 - t5)
        metrics['perf_update_total_ms']        = 1000.0 * (t6 - t0)

        return metrics