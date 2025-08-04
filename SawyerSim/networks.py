import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

from MAE_Model.model import MAEModel

class CriticNetwork(nn.Module):
    def __init__(self, beta, critic_input_dim, n_actions, fc1_dims=256, fc2_dims=256,
            name='critic', mvmae : torch.nn.Module|None = None, smaller_encoder_embed_dim=128):
        super(CriticNetwork, self).__init__()
        self.critic_input_dim = critic_input_dim
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name

        self.fc1 = nn.Linear(self.critic_input_dim + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.mvmae = mvmae
        self.z_projection = nn.Linear(mvmae.encoder_embed_dim, smaller_encoder_embed_dim)
        self.to(self.device)

    def forward(self, observation, action):
        with torch.no_grad():
            _, _, z = self.mvmae(torch.tensor(observation["image_observation"], dtype=torch.float32))
        
        z = self.z_projection(z)
        z_flat = z.view(z.size(0), -1)
        state = torch.tensor(observation["state_observation"], dtype=torch.float32)
        
        action_value = self.fc1(torch.cat([z_flat, state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)
        return q

class ActorNetwork(nn.Module):
    def __init__(self, alpha, actor_input_dim, max_action=None, fc1_dims=256, 
            fc2_dims=256, n_actions=None, name='actor', mvmae : torch.nn.Module|None = None,
            smaller_encoder_embed_dim=128):
        
        super(ActorNetwork, self).__init__()
        self.actor_input_dim = actor_input_dim
        self.n_actions = n_actions
        self.name = name
        self.max_action = max_action
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(self.actor_input_dim, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mu = nn.Linear(fc2_dims, self.n_actions)
        self.sigma = nn.Linear(fc2_dims, self.n_actions)
        
        self.mvmae = mvmae
        self.z_projection = nn.Linear(mvmae.encoder_embed_dim, smaller_encoder_embed_dim)
        self.out, self.mask = None, None # Used for calculating reconstruction loss
        
        self.optimizer = optim.Adam(list(self.parameters()) + list(mvmae.parameters()), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        prob = F.relu(self.fc1(x))
        prob = F.relu(self.fc2(prob))

        mu = self.mu(prob)
        sigma = self.sigma(prob)
        sigma = torch.clamp(sigma, min=self.reparam_noise, max=1)
        return mu, sigma
 
    def sample_normal(self, observation, reparameterize=True):
        """
        Args:
            observation (dict): 
                image observation: a single tensor of shape (batch, height, width_total, channels), for passing into mvmae
                state observation: a flat numpy vector of n elements, tensor of shape (batch, n)
            reparameterize (bool, optional): _description_. Defaults to True.
        """
        out, mask, z = self.mvmae(torch.tensor(observation["image_observation"], dtype=torch.float32))
        z = self.z_projection(z)
        z_flat = z.view(z.size(0), -1)
        state = torch.tensor(observation["state_observation"], dtype=torch.float32)
        
        mu, sigma = self.forward(torch.cat([z_flat, state], dim=1))
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = torch.tanh(actions)*torch.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= torch.log(1-action.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs, out, mask