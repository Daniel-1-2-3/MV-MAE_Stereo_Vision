import os
import torch
import torch.nn.functional as F
import numpy as np
from SawyerSim.buffer import ReplayBuffer
from SawyerSim.networks import ActorNetwork, CriticNetwork
from MAE_Model.model import MAEModel

class Agent():
    def __init__(self, 
        env=None, 
        alpha=0.0003, # Learning rate
        beta=0.0003,
        gamma=0.99, # Discount factor
        tau=0.005, # Update rate
        buffer_size=5000, # Size for replay buffer
        nviews=2,
        patch_size=8,
        encoder_embed_dim=768,
        smaller_encoder_embed_dim=128, # Make embed size smaller, for memory issues
        decoder_embed_dim=512,
        encoder_heads=16,
        decoder_heads=16,
        in_channels=3,
        img_h_size=128,
        img_w_size=128, 
        batch_size=32,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        # Hyperparams for MVMAE
        self.nviews = nviews
        self.patch_size = patch_size
        self.encoder_embed_dim = encoder_embed_dim
        self.smaller_encoder_embed_dim = smaller_encoder_embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.encoder_heads = encoder_heads
        self.decoder_heads = decoder_heads
        self.in_channels = in_channels
        self.img_h_size = img_h_size
        self.img_w_size = img_w_size
        self.batch_size = batch_size
        
        self.n_actions = env.action_space.shape[0]
        # Input dim into the actor, a concat of flattened encoder output & states (array of 18)
        self.z_num_patches = (self.img_h_size * self.img_w_size) / (self.patch_size ** 2) * self.nviews # Number of patches in the encoder output
        self.input_dim = int(self.z_num_patches) * int(self.smaller_encoder_embed_dim) + int(env.observation_space['state_observation'].shape[0])

        self.mvmae = MAEModel(
            nviews=self.nviews,
            patch_size=self.patch_size,
            encoder_embed_dim=self.encoder_embed_dim,
            decoder_embed_dim=self.decoder_embed_dim,
            encoder_heads=self.encoder_heads,
            decoder_heads=self.decoder_heads,
            in_channels=self.in_channels,
            img_h_size=self.img_h_size,
            img_w_size=self.img_w_size
        )
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mvmae.to(self.device)
        self.mvmae.train()  
        
        self.memory = ReplayBuffer(self.buffer_size, 
                                   env.observation_space["image_observation"].shape, 
                                   env.observation_space["state_observation"].shape, self.n_actions)
        self.actor = ActorNetwork(self.alpha, self.input_dim, n_actions=self.n_actions, name='actor', max_action=env.action_space.high, mvmae=self.mvmae)
        self.critic_1 = CriticNetwork(self.beta, self.input_dim, n_actions=self.n_actions, name='critic_1', mvmae=self.mvmae)
        self.critic_2 = CriticNetwork(self.beta, self.input_dim, n_actions=self.n_actions, name='critic_2', mvmae=self.mvmae)
        # Two critics are often used to avoid overestimation

    def choose_action(self, observation):
        actions, _, _, _ = self.actor.sample_normal(observation, reparameterize=False)
        return actions.cpu().detach().numpy()[0]

    def remember(self, obs, action, reward, new_obs, done):
        self.memory.store_transition(obs, action, reward, new_obs, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        # Sample from the replay buffer
        obs, action, reward, obs_, done = self.memory.sample_buffer(self.batch_size)
        obs = {
            "image_observation": torch.as_tensor(obs["image_observation"], dtype=torch.float32, device=self.device),
            "state_observation": torch.as_tensor(obs["state_observation"], dtype=torch.float32, device=self.device)
        }
        obs_ = {
            "image_observation": torch.as_tensor(obs_["image_observation"], dtype=torch.float32, device=self.device),
            "state_observation": torch.as_tensor(obs_["state_observation"], dtype=torch.float32, device=self.device)
        }
        print(f"[DEBUG] image_observation AFTER SAMPLING FROM BUFFER : min={obs["image_observation"].min().item():.4f}, max={obs.max().item():.4f}")
       
        reward = torch.as_tensor(reward, dtype=torch.float).to(self.actor.device)
        done = torch.as_tensor(done).to(self.actor.device)
        action = torch.as_tensor(action, dtype=torch.float).to(self.actor.device)

        # Step the actor, reparameterized allows for gradient flow
        actions, log_probs, out, mask = self.actor.sample_normal(obs, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(obs, actions)
        q2_new_policy = self.critic_2.forward(obs, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        
        policy_loss = torch.mean(self.alpha * log_probs - critic_value)
        target = torch.as_tensor(obs["image_observation"], dtype=torch.float32, device=self.actor.device)
        
        print(f"[DEBUG] out: min={out.min().item():.4f}, max={out.max().item():.4f}")
        print(f"[DEBUG] target: min={target.min().item():.4f}, max={target.max().item():.4f}")

        mvmae_loss = self.mvmae.compute_loss(out, target, mask)
        loss = policy_loss + mvmae_loss
        
        self.actor.optimizer.zero_grad()
        loss.backward()
        self.actor.optimizer.step()

        # Step the critic
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        with torch.no_grad():
            next_actions, next_log_probs, _, _ = self.actor.sample_normal(obs_, reparameterize=False)
            next_q1 = self.critic_1(obs_, next_actions)
            next_q2 = self.critic_2(obs_, next_actions)
            next_q = torch.min(next_q1, next_q2).view(-1)
            next_q -= self.alpha * next_log_probs.view(-1)
            q_hat = reward + self.gamma * next_q * (1 - done.float())
        
        q1_old_policy = self.critic_1.forward(obs, action).view(-1)
        q2_old_policy = self.critic_2.forward(obs, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()
        
        print('MVMAE_Loss:', mvmae_loss.item(), 'Policy_Loss', policy_loss.item())