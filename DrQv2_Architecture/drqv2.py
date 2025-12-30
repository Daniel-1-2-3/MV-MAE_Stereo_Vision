import numpy as np
import time
import jax
import jax.numpy as jp
import torch
import torch.nn as nn
import torch.nn.functional as F

import DrQv2_Architecture.utils as utils
from MAE_Model.model import MAEModel
from MAE_Model.prepare_input import Prepare
from DrQv2_Architecture.replay_buffer import ReplayBuffer
from DrQv2_Architecture.video import VideoRecorder
from gymnasium.spaces import Box

class Actor(nn.Module):
    def __init__(self, repr_dim, action_dim, feature_dim, hidden_dim):
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
            nn.Linear(hidden_dim, action_dim)
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
    def __init__(self, repr_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2
class DrQv2Agent:
    def __init__(self,
        # General variables
        env: None,
        num_envs: int,
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
        # Actor and critic
        feature_dim: int = 100,
        hidden_dim: int = 1024,
        discount: float = 0.99,
        # RL variables
        buffer_size: int = 100_000,
        sample_batch_size: int = 256,
        episode_length: int = 300,
        critic_target_tau: float = 0.001, # Soft-update for target critic
        num_expl_steps: int = 2000, 
        update_every_steps: int = 2,
        update_mvmae_every_steps: int = 10,
        stddev_schedule: str = 'linear(1.0,0.1,500000)', # Type of scheduler, value taken from cfgs/task/medium.yaml, stddev for exploration noise
        stddev_clip: int = 0.3, # How much to clip sampled action noise
    ):
        self.env = env
        self.num_envs = num_envs
        self.action_space = Box(low = -1.0, high = 1.0, shape = (8,), dtype = np.float32)
        self.action_dim = self.action_space.shape[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self.discount = discount
        
        self.buffer_size = buffer_size
        self.sample_batch_size = sample_batch_size
        self.episode_length = episode_length
        self.critic_target_tau = critic_target_tau
        self.num_expl_steps = num_expl_steps
        self.update_every_steps = update_every_steps
        self.update_mvmae_every_steps = update_mvmae_every_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        
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
        
        self.actor = Actor(self.repr_dim, self.action_dim, self.feature_dim, self.hidden_dim).to(self.device)
        self.critic = Critic(self.repr_dim, self.action_dim, feature_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(self.repr_dim, self.action_dim, feature_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.mvmae_optim = torch.optim.Adam(self.mvmae.parameters(), lr=lr)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            self.buffer_size, (self.img_h_size, 2 * self.img_w_size, self.in_channels), self.action_dim, self.device
        )

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
        with torch.no_grad():
            obs = Prepare.normalize(obs)
            z, _ = self.mvmae.encoder(obs, mask_x=False)
            obs_latent = z.flatten(start_dim=-2)

            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(obs_latent, stddev)

            if eval_mode:
                action = dist.mean
            else:
                action = dist.sample(clip=None)
                if step < self.num_expl_steps:
                    action.uniform_(-1.0, 1.0)
        
        return action
    
    def update_critic(self, z, action, reward, discount, z_next, step, obs, update_mvmae: bool):
        metrics = dict()

        # Calculates target, computes MSE critic loss
        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(z_next, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(z_next, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)
        
        # Critic forward
        Q1, Q2 = self.critic(z, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        # MV-MAE reconstruction inside critic update
        if update_mvmae:
            out, mask, _ = self.mvmae.forward(obs, mask_x=True)
            recon_loss = self.mvmae.compute_loss(out, obs, mask)

        total_loss = critic_loss
        if update_mvmae:
            total_loss += self.coef_mvmae * recon_loss

        # Backpropagate critic + mvmae losses together
        self.critic_optim.zero_grad(set_to_none=True)
        if update_mvmae:
            self.mvmae_optim.zero_grad(set_to_none=True)
        
        total_loss.backward()
        
        self.critic_optim.step()
        if update_mvmae:
            self.mvmae_optim.step()

        metrics['critic_target_q'] = target_Q.mean().item()
        metrics['critic_q1'] = Q1.mean().item()
        metrics['critic_q2'] = Q2.mean().item()
        metrics['critic_loss'] = critic_loss.item()
        metrics['recon_loss'] = recon_loss.item() if update_mvmae else -1.0
        metrics['total_loss'] = total_loss.item()

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

        metrics['actor_loss'] = actor_loss.item()
        metrics['actor_logprob'] = log_prob.mean().item()
        metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics
    
    def update(self, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics
        update_mvmae = True if step % self.update_mvmae_every_steps == 0 else False

        # Sample a batch from the replay buffer
        batch = self.replay_buffer.sample(self.sample_batch_size)
        obs, action, reward, discount, next_obs, done = utils.to_torch(batch, self.device)
        metrics['batch_reward'] = reward.mean().item()

        obs = Prepare.normalize(obs)
        next_obs = Prepare.normalize(next_obs)

        # Encode obs with gradients on only every 10 steps, no grads otherwise
        z = None
        if (update_mvmae):
            z, _ = self.mvmae.encoder(obs, mask_x=False)
            z = z.flatten(start_dim=-2)
        else:
            with torch.no_grad():
                z, _ = self.mvmae.encoder(obs, mask_x=False)
                z = z.flatten(start_dim=-2)

        # Encode next_obs with gradients off
        with torch.no_grad():
            z_next, _ = self.mvmae.encoder(next_obs, mask_x=False)
            z_next = z_next.flatten(start_dim=-2)

        # Critic + MV-MAE recon, Actor
        metrics_c = self.update_critic(z, action, reward, discount, z_next, step, obs, update_mvmae)
        metrics_a = self.update_actor(z.detach(), step)

        metrics.update(metrics_c)
        metrics.update(metrics_a)
        
        # Soft update target critic (Polyak averaging)
        with torch.no_grad():
            tau = float(self.critic_target_tau)
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.mul_(1.0 - tau).add_(tau * p.data)
                
        return metrics
    
    def learn(self, total_timesteps: int, learning_starts: int) -> None:
        self.train(training=True)
        global_step_count, episode_step_count = 0, 0
        episode_reward = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        num_episodes = 0
        
        obs_td = self.env.reset() # TensorDict, batch=[num_envs]
        obs = obs_td["state"] # torch.Tensor on GPU (via dlpack)
        
        while global_step_count <= total_timesteps:
            if episode_step_count >= self.episode_length:
                print("Step:", global_step_count, "Mean ep reward:", (episode_reward.mean().item() / self.episode_length))
                num_episodes += 1

                obs_td = self.env.reset() # Reset env (wrapper API)
                obs = obs_td["state"]

                episode_step_count = 0
                episode_reward = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
                
            with torch.no_grad():
                action = self.act(obs, global_step_count, eval_mode=False)

            if global_step_count > learning_starts:
                metrics = self.update(global_step_count)

            next_obs_td, rew_t, done_t, info = self.env.step(action)
            next_obs = next_obs_td["state"]  # torch.Tensor uint8
            
            rew_t = rew_t.reshape(-1).to(torch.float32)
            done_t = done_t.reshape(-1).to(torch.bool)

            disc_t = (1.0 - done_t.float()) * float(self.discount)
            self.replay_buffer.add_batch(obs, action, rew_t, disc_t, next_obs, done_t)

            obs = next_obs
            episode_reward = episode_reward + rew_t
            episode_step_count += 1
            global_step_count += 1

            # Early terminate/reset on done.
            # Wrapper only supports full-batch reset(), so if ANY env is done, reset all.
            if done_t.any():
                print("Mean ep reward:", (episode_reward.mean().item() / max(1, episode_step_count)))
                num_episodes += 1

                obs_td = self.env.reset()
                obs = obs_td["state"]

                episode_step_count = 0
                episode_reward = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)

